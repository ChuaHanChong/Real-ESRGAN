"""Standalone port of RealESRGANModel.feed_data() — degrade a folder of images.

Defaults match options/finetune_realesrgan_x4plus.yml. USM-on-GT off, no crop, no queue.
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from basicsr.data.degradations import (circular_lowpass_kernel, random_add_gaussian_noise_pt,
                                       random_add_poisson_noise_pt, random_mixed_kernels)
from basicsr.utils.diffjpeg import DiffJPEG
from basicsr.utils.img_process_util import USMSharp, filter2D


# ---------------------------------------------------------------------------
# Degradation config
# ---------------------------------------------------------------------------
# Defaults mirror options/finetune_realesrgan_x4plus.yml — the standard
# Real-ESRGAN paper recipe. Tune these for your dataset.
@dataclass
class DegradationConfig:
    # First degradation pass
    resize_prob: List[float] = field(default_factory=lambda: [0.2, 0.7, 0.1])  # up, down, keep
    resize_range: Tuple[float, float] = (0.15, 1.5)
    gaussian_noise_prob: float = 0.5
    noise_range: Tuple[float, float] = (1, 30)
    poisson_scale_range: Tuple[float, float] = (0.05, 3.0)
    gray_noise_prob: float = 0.4
    jpeg_range: Tuple[int, int] = (30, 95)

    # Second degradation pass
    second_blur_prob: float = 0.8
    resize_prob2: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    resize_range2: Tuple[float, float] = (0.3, 1.2)
    gaussian_noise_prob2: float = 0.5
    noise_range2: Tuple[float, float] = (1, 25)
    poisson_scale_range2: Tuple[float, float] = (0.05, 2.5)
    gray_noise_prob2: float = 0.4
    jpeg_range2: Tuple[int, int] = (30, 95)

    # Kernel sampling (used when generating kernel1 / kernel2 / sinc kernel)
    blur_kernel_size: int = 21
    kernel_list: Tuple[str, ...] = (
        'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso')
    kernel_prob: Tuple[float, ...] = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)
    sinc_prob: float = 0.1
    blur_sigma: Tuple[float, float] = (0.2, 3.0)
    betag_range: Tuple[float, float] = (0.5, 4.0)
    betap_range: Tuple[float, float] = (1.0, 2.0)

    blur_kernel_size2: int = 21
    kernel_list2: Tuple[str, ...] = (
        'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso')
    kernel_prob2: Tuple[float, ...] = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)
    sinc_prob2: float = 0.1
    blur_sigma2: Tuple[float, float] = (0.2, 1.5)
    betag_range2: Tuple[float, float] = (0.5, 4.0)
    betap_range2: Tuple[float, float] = (1.0, 2.0)

    final_sinc_prob: float = 0.8

    # Pipeline switches — defaults match finetune_realesrgan_x4plus.yml exactly.
    use_usm_on_gt: bool = True  # YAML feed_data uses USM(GT) as input to degradation.
    scale: int = 4              # Output size = (H // scale, W // scale).


# ---------------------------------------------------------------------------
# Kernel sampling — ported from realesrgan/data/realesrgan_dataset.py
# ---------------------------------------------------------------------------
KERNEL_RANGE = [2 * v + 1 for v in range(3, 11)]  # 7, 9, 11, ..., 21
PULSE_KERNEL = torch.zeros(21, 21).float()
PULSE_KERNEL[10, 10] = 1


def _sample_kernel(cfg_size: int, cfg_kernel_list, cfg_kernel_prob, cfg_sinc_prob,
                   cfg_blur_sigma, cfg_betag_range, cfg_betap_range) -> np.ndarray:
    kernel_size = random.choice(KERNEL_RANGE)
    if np.random.uniform() < cfg_sinc_prob:
        omega_c = (np.random.uniform(np.pi / 3, np.pi)
                   if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi))
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            list(cfg_kernel_list), list(cfg_kernel_prob), kernel_size,
            list(cfg_blur_sigma), list(cfg_blur_sigma), [-math.pi, math.pi],
            list(cfg_betag_range), list(cfg_betap_range), noise_range=None)
    pad_size = (21 - kernel_size) // 2
    return np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))


def sample_kernels_for_image(cfg: DegradationConfig):
    k1 = _sample_kernel(cfg.blur_kernel_size, cfg.kernel_list, cfg.kernel_prob,
                        cfg.sinc_prob, cfg.blur_sigma, cfg.betag_range, cfg.betap_range)
    k2 = _sample_kernel(cfg.blur_kernel_size2, cfg.kernel_list2, cfg.kernel_prob2,
                        cfg.sinc_prob2, cfg.blur_sigma2, cfg.betag_range2, cfg.betap_range2)
    if np.random.uniform() < cfg.final_sinc_prob:
        kernel_size = random.choice(KERNEL_RANGE)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc = torch.FloatTensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=21))
    else:
        sinc = PULSE_KERNEL.clone()
    return torch.FloatTensor(k1), torch.FloatTensor(k2), sinc


# ---------------------------------------------------------------------------
# The degradation pipeline — ported from RealESRGANModel.feed_data
# ---------------------------------------------------------------------------
class Degrader:
    def __init__(self, cfg: DegradationConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.jpeger = DiffJPEG(differentiable=False).to(device)
        self.usm = USMSharp().to(device) if cfg.use_usm_on_gt else None

    @torch.no_grad()
    def degrade(self, gt: torch.Tensor, kernel1: torch.Tensor, kernel2: torch.Tensor,
                sinc_kernel: torch.Tensor) -> torch.Tensor:
        """gt: (B, 3, H, W) in [0, 1]. Returns degraded LQ at (H/scale, W/scale)."""
        cfg = self.cfg
        gt = gt.to(self.device)
        kernel1 = kernel1.to(self.device)
        kernel2 = kernel2.to(self.device)
        sinc_kernel = sinc_kernel.to(self.device)

        ori_h, ori_w = gt.shape[2:4]
        out = self.usm(gt) if self.usm is not None else gt

        # ---- First degradation ----
        out = filter2D(out, kernel1)
        updown = random.choices(['up', 'down', 'keep'], cfg.resize_prob)[0]
        scale_factor = (np.random.uniform(1, cfg.resize_range[1]) if updown == 'up'
                        else np.random.uniform(cfg.resize_range[0], 1) if updown == 'down' else 1)
        out = F.interpolate(out, scale_factor=scale_factor,
                            mode=random.choice(['area', 'bilinear', 'bicubic']))
        if np.random.uniform() < cfg.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=list(cfg.noise_range), clip=True, rounds=False,
                gray_prob=cfg.gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out, scale_range=list(cfg.poisson_scale_range), gray_prob=cfg.gray_noise_prob,
                clip=True, rounds=False)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ---- Second degradation ----
        if np.random.uniform() < cfg.second_blur_prob:
            out = filter2D(out, kernel2)
        updown = random.choices(['up', 'down', 'keep'], cfg.resize_prob2)[0]
        scale_factor = (np.random.uniform(1, cfg.resize_range2[1]) if updown == 'up'
                        else np.random.uniform(cfg.resize_range2[0], 1) if updown == 'down' else 1)
        out = F.interpolate(
            out,
            size=(int(ori_h / cfg.scale * scale_factor), int(ori_w / cfg.scale * scale_factor)),
            mode=random.choice(['area', 'bilinear', 'bicubic']))
        if np.random.uniform() < cfg.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=list(cfg.noise_range2), clip=True, rounds=False,
                gray_prob=cfg.gray_noise_prob2)
        else:
            out = random_add_poisson_noise_pt(
                out, scale_range=list(cfg.poisson_scale_range2), gray_prob=cfg.gray_noise_prob2,
                clip=True, rounds=False)

        # ---- Final: [resize-back + sinc] and JPEG, in random order ----
        target_h, target_w = ori_h // cfg.scale, ori_w // cfg.scale
        if np.random.uniform() < 0.5:
            out = F.interpolate(out, size=(target_h, target_w),
                                mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            out = F.interpolate(out, size=(target_h, target_w),
                                mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, sinc_kernel)

        return torch.clamp(out, 0, 1)


# ---------------------------------------------------------------------------
# I/O helpers — handle 1-channel infrared by replicating to 3 channels
# ---------------------------------------------------------------------------
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_image_as_rgb_tensor(path: Path) -> Tuple[torch.Tensor, bool]:
    """Returns (tensor (1,3,H,W) in [0,1], was_grayscale)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Failed to read {path}')
    was_gray = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]  # drop alpha
    img = img.astype(np.float32) / 255.0  # BGR float32
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, was_gray


def save_tensor(out: torch.Tensor, path: Path, was_gray: bool):
    img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,3) BGR float32
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if was_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    # PNG to avoid stacking another JPEG round-trip on top of the simulated JPEG.
    if path.suffix.lower() in {'.jpg', '.jpeg'}:
        path = path.with_suffix('.png')
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scale', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=None,
                   help='Only process first N images (smoke test).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--use-usm', action='store_true',
                   help='Apply USM sharpening to GT before degradation (training-style).')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = DegradationConfig(scale=args.scale, use_usm_on_gt=args.use_usm)
    device = torch.device(args.device)
    degrader = Degrader(cfg, device)

    images = collect_images(args.input)
    if args.limit:
        images = images[:args.limit]
    print(f'Found {len(images)} images under {args.input}')

    for i, src in enumerate(images):
        rel = src.relative_to(args.input)
        dst = args.output / rel
        gt, was_gray = load_image_as_rgb_tensor(src)
        k1, k2, sinc = sample_kernels_for_image(cfg)
        out = degrader.degrade(gt, k1.unsqueeze(0), k2.unsqueeze(0), sinc.unsqueeze(0))
        save_tensor(out, dst, was_gray)
        if (i + 1) % 25 == 0 or i + 1 == len(images):
            print(f'  [{i + 1}/{len(images)}] {rel}')

    print(f'Done. Output -> {args.output}')


if __name__ == '__main__':
    sys.exit(main())
