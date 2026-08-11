"""Prove that DegradationConfig() defaults match options/finetune_realesrgan_x4plus.yml.

Loads the YAML, walks every field of DegradationConfig, prints a side-by-side table,
and exits non-zero if any mismatch is found.
"""

import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade_dataset import DegradationConfig  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
YAML_PATH = REPO / 'options' / 'finetune_realesrgan_x4plus.yml'

# Field name in DegradationConfig -> (yaml path as list of keys)
# Top-level keys in the YAML are either at root or under datasets.train.
ROOT_KEYS = {
    'scale',
    'resize_prob', 'resize_range', 'gaussian_noise_prob', 'noise_range',
    'poisson_scale_range', 'gray_noise_prob', 'jpeg_range', 'second_blur_prob',
    'resize_prob2', 'resize_range2', 'gaussian_noise_prob2', 'noise_range2',
    'poisson_scale_range2', 'gray_noise_prob2', 'jpeg_range2',
}
TRAIN_KEYS = {
    'blur_kernel_size', 'kernel_list', 'kernel_prob', 'sinc_prob',
    'blur_sigma', 'betag_range', 'betap_range',
    'blur_kernel_size2', 'kernel_list2', 'kernel_prob2', 'sinc_prob2',
    'blur_sigma2', 'betag_range2', 'betap_range2',
    'final_sinc_prob',
}
# `use_usm_on_gt` is implied by realesrgan_model.py:83 (uses gt_usm) — not a YAML key
# but a hard-coded behavior of the GAN model. We treat it as derived: must be True.
DERIVED_TRUE = {'use_usm_on_gt'}


def normalize(v):
    """Cast to a comparable form: tuples/lists -> list of float; numbers -> float."""
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def main():
    yml = yaml.safe_load(YAML_PATH.read_text())
    train = yml['datasets']['train']
    cfg = DegradationConfig()

    rows = []
    mismatches = 0
    for f in fields(cfg):
        name = f.name
        cfg_v = getattr(cfg, name)
        if name in DERIVED_TRUE:
            ok = (cfg_v is True)
            rows.append((name, '(derived from feed_data: True)', repr(cfg_v),
                         'MATCH' if ok else 'MISMATCH'))
            if not ok:
                mismatches += 1
            continue
        if name in ROOT_KEYS:
            yaml_v = yml.get(name)
        elif name in TRAIN_KEYS:
            yaml_v = train.get(name)
        else:
            rows.append((name, '???', repr(cfg_v), 'UNCLASSIFIED'))
            mismatches += 1
            continue
        ok = normalize(yaml_v) == normalize(cfg_v)
        rows.append((name, repr(yaml_v), repr(cfg_v), 'MATCH' if ok else 'MISMATCH'))
        if not ok:
            mismatches += 1

    name_w = max(len(r[0]) for r in rows)
    yaml_w = max(len(r[1]) for r in rows)
    cfg_w  = max(len(r[2]) for r in rows)
    sep = '-' * (name_w + yaml_w + cfg_w + 14)
    print(sep)
    print(f'{"field":<{name_w}}  {"yaml":<{yaml_w}}  {"DegradationConfig":<{cfg_w}}  status')
    print(sep)
    for r in rows:
        print(f'{r[0]:<{name_w}}  {r[1]:<{yaml_w}}  {r[2]:<{cfg_w}}  {r[3]}')
    print(sep)
    print(f'{mismatches} mismatch(es) out of {len(rows)} fields')
    return 0 if mismatches == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
