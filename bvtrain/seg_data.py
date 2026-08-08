"""Segmentation data loading: images + pseudo-mask labels.

Loads a companion HF dataset (image + binary mask + species) built offline by
``scripts/generate_pseudo_masks.py``. The mask is a single-channel PNG (0 =
background, 255 = plant). Used to train a distilled student segmenter that the
demo applies as a soft-mask preprocessor before the ResNet-50 classifier.

The train transform is a *paired* callable — anything spatial (crop, flip,
resize) has to run on the image and mask together, so this module deliberately
does NOT accept a torchvision ``transforms.Compose``. Notebook 06 defines the
train transform inline (keeping the "customize in the notebook" convention from
CLAUDE.md); this module ships only the deterministic eval transform.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .data import IMG_SIZE, MEAN, STD


def eval_seg_transforms(size: int = IMG_SIZE):
    """Deterministic Resize(256) + CenterCrop(size), applied to image and mask
    together. Image is normalized with ImageNet stats; mask becomes 1xHxW float
    in {0, 1}. Matches the classifier's eval transform in ``bvtrain/data.py``."""

    def _paired(img, mask):
        img = TF.resize(img, 256)
        mask = TF.resize(mask, 256, interpolation=InterpolationMode.NEAREST)
        img = TF.center_crop(img, size)
        mask = TF.center_crop(mask, size)
        img = TF.to_tensor(img)
        img = TF.normalize(img, MEAN, STD)
        m = torch.from_numpy(np.array(mask, dtype=np.uint8))
        m = (m > 127).float().unsqueeze(0)
        return img, m

    return _paired


@dataclass
class SegData:
    counts: dict            # {"train": N, "val": N, "test": N}
    _hf: object             # HuggingFace DatasetDict with image + mask + species


def load_seg_data(env, masks_repo: str | None = None) -> SegData:
    """Load the companion masks dataset from Hugging Face."""
    from datasets import load_dataset
    repo = masks_repo or getattr(env, "hf_masks_repo", None)
    if not repo:
        raise SystemExit(
            "No masks repo. Pass masks_repo=... or set env.hf_masks_repo. "
            "Generate one first: python scripts/generate_pseudo_masks.py --repo <user>/botanical-vision-256-masks"
        )
    hf = load_dataset(repo)
    counts = {k: len(hf[k]) for k in hf}
    print(f"seg dataset: {counts}")
    return SegData(counts, hf)


class _SegSet(Dataset):
    """Wraps an HF split; ``tf`` is a paired callable (PIL, PIL) -> (tensor, tensor)."""

    def __init__(self, ds, tf):
        self.ds = ds
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        mask = ex["mask"].convert("L")
        return self.tf(img, mask)


@dataclass
class SegLoaders:
    train_ds: object
    val: object           # DataLoader
    test: object          # DataLoader
    batch: int
    accum: int
    num_workers: int
    use_amp: bool
    train_tf: object


def build_seg_loaders(data: SegData, train_tf, env, eval_tf=None,
                      effective_batch: int = 32) -> SegLoaders:
    """Build val/test loaders + a train dataset with a VRAM-sized batch.

    Segmentation models are heavier per-sample than the ResNet-50 classifier
    (dense feature maps for the decoder), so the per-VRAM batch is roughly half
    what ``bvtrain.data.build_loaders`` picks — the effective batch stays
    matched via gradient accumulation.
    """
    eval_tf = eval_tf or eval_seg_transforms()
    train_ds = _SegSet(data._hf["train"], train_tf)
    val_ds = _SegSet(data._hf["val"], eval_tf)
    test_ds = _SegSet(data._hf["test"], eval_tf)

    dev = env.device
    if dev.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch = 32 if vram > 12 else 16 if vram > 8 else 8 if vram > 6 else 4
    else:
        batch = 8
    num_workers = 0 if os.name == "nt" else 2
    use_amp = dev.type == "cuda"
    accum = max(1, round(effective_batch / batch))

    val = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    test = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    print(f"BATCH {batch} x ACCUM {accum} = eff {batch*accum} | workers {num_workers} | amp {use_amp}")
    return SegLoaders(train_ds, val, test, batch, accum, num_workers, use_amp, train_tf)
