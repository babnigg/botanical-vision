"""Data loading: species labels, the train/val/test datasets, and dataloaders.

Reads local `../data` files if present (maintainer), else the streamed HF dataset
(Colab / Kaggle / teammates). `build_loaders` sizes the batch to the GPU and picks a
gradient-accumulation factor so the *effective* batch is constant across machines.
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]   # ImageNet normalization — matches the pretrained ResNet-50
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def eval_transforms():
    """The standard ImageNet eval transform (Resize 256 / CenterCrop 224 / normalize)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@dataclass
class Data:
    labels: list
    label2idx: dict
    n_species: object            # the subset toggle value (None or int), kept for the run signature
    _local_splits: object = None  # a pandas DataFrame, or None on the HF path
    _hf: object = None

    @property
    def n_labels(self) -> int:
        return len(self.labels)


def load_data(env, n_species=None, data_dir: str = "../data") -> Data:
    if env.use_local:
        import pandas as pd
        splits = pd.read_csv(f"{data_dir}/splits.csv")
        if n_species:
            keep = sorted(splits["species"].unique())[:n_species]
            splits = splits[splits["species"].isin(keep)].reset_index(drop=True)
        labels = sorted(splits["species"].unique())
        counts = splits["split"].value_counts().to_dict()
        data = Data(labels, {s: i for i, s in enumerate(labels)}, n_species, _local_splits=splits)
    else:
        from datasets import load_dataset
        hf = load_dataset(env.hf_repo)
        if n_species:
            keep = set(sorted(hf["train"].features["label"].names)[:n_species])
            hf = hf.filter(lambda e: e["species"] in keep)
            labels = sorted(set(hf["train"]["species"]))
        else:
            labels = sorted(hf["train"].features["label"].names)
        counts = {k: len(hf[k]) for k in hf}
        data = Data(labels, {s: i for i, s in enumerate(labels)}, n_species, _hf=hf)
    print(f"{data.n_labels} species | {counts}")
    return data


def _apply_bg_blur(img, mask, blur_radius: int = 15):
    """Blend a Gaussian-blurred background into ``img`` using ``mask`` (L-mode PIL,
    0 = background, 255 = plant). Returns ``img`` unchanged when the mask covers
    essentially everything (nothing to blur). Called by both the local and HF
    datasets so the recipe stays identical across paths.
    """
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.BILINEAR)  # match publisher's resample filter
    m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    if m.mean() > 0.98:            # all-plant mask = nothing to blur
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
    out = np.asarray(img) * m + np.asarray(blurred) * (1 - m)
    return Image.fromarray(np.rint(out).astype("uint8"))


class _PlantSetLocal(Dataset):
    def __init__(self, df, tf, label2idx, mask_dir=None, bg_aug_p=0.0):
        self.df = df.reset_index(drop=True)
        self.tf = tf
        self.l = label2idx
        # background-blur augmentation: with prob bg_aug_p, gaussian-blur the
        # background using a precomputed plant mask (same soft-mask the demo
        # applies), so background-invariance is learned in-distribution.
        # masks live in mask_dir as <md5-of-path>.png (scripts/precompute_student_masks.py)
        self.mask_dir = mask_dir
        self.bg_aug_p = bg_aug_p

    def __len__(self):
        return len(self.df)

    def _bg_blur(self, img, path):
        if self.mask_dir is None or random.random() >= self.bg_aug_p:
            return img
        mp = self.mask_dir / (hashlib.md5(str(path).encode()).hexdigest() + ".png")
        if not mp.exists():
            return img
        return _apply_bg_blur(img, Image.open(mp).convert("L"))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # cloud-synced local data (OneDrive) can throw transient read errors under load
        for attempt in range(3):
            try:
                img = Image.open(r["path"]).convert("RGB")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        img = self._bg_blur(img, r["path"])
        return self.tf(img), self.l[r["species"]]


class _PlantSetHF(Dataset):
    def __init__(self, ds, tf, label2idx, bg_aug_p=0.0):
        self.ds = ds
        self.tf = tf
        self.l = label2idx
        # bg-aug on the HF path only fires when the underlying dataset has a
        # "mask" field (i.e. it's the paired masks-companion dataset from
        # scripts/publish_student_masks_full.py). No field, no cost — the
        # normal source dataset short-circuits before touching the RNG.
        self.bg_aug_p = bg_aug_p
        self._has_mask = "mask" in ds.features

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        if self._has_mask and self.bg_aug_p > 0 and random.random() < self.bg_aug_p:
            img = _apply_bg_blur(img, ex["mask"].convert("L"))
        return self.tf(img), self.l[ex["species"]]


def _make_ds(data: Data, split: str, tf, mask_dir=None, mask_repo=None, bg_aug_p=0.0):
    if data._local_splits is not None:
        return _PlantSetLocal(data._local_splits[data._local_splits["split"] == split], tf, data.label2idx,
                              mask_dir=mask_dir if split == "train" else None,
                              bg_aug_p=bg_aug_p if split == "train" else 0.0)
    # HF: for the train split with a masks-companion repo, load the paired
    # {image, mask, species, split} dataset in place of the source images so
    # _PlantSetHF can apply bg-aug per-sample. Val/test keep the source (eval
    # is always on natural backgrounds).
    if split == "train" and mask_repo:
        from datasets import load_dataset
        hf_masks = load_dataset(mask_repo, split="train")
        if "mask" not in hf_masks.features:
            raise ValueError(
                f"mask_repo {mask_repo!r} has no 'mask' field "
                f"(features: {list(hf_masks.features)}) — expected the paired "
                "companion dataset from scripts/publish_student_masks_full.py"
            )
        if data.n_species:
            keep = set(data.labels)
            hf_masks = hf_masks.filter(lambda e: e["species"] in keep)
        return _PlantSetHF(hf_masks, tf, data.label2idx, bg_aug_p=bg_aug_p)
    return _PlantSetHF(data._hf[split], tf, data.label2idx)


@dataclass
class Loaders:
    train_ds: object
    val: object          # DataLoader
    test: object         # DataLoader
    batch: int
    accum: int
    num_workers: int
    use_amp: bool
    train_tf: object
    labels: list
    n_species: object
    bg_aug_p: float = 0.0
    mask_key: str = ""    # "local:<dir>" or "hf:<repo>" — recorded so _signature() can split runs

    @property
    def n_labels(self) -> int:
        return len(self.labels)


def build_loaders(data: Data, train_tf, env, eval_tf=None, effective_batch: int = 64,
                  mask_dir=None, mask_repo=None, bg_aug_p: float = 0.0) -> Loaders:
    """Build val/test loaders + a train dataset, with a VRAM-sized batch.

    Small cards (a 4 GB laptop GPU) don't OOM on Windows — they silently spill to shared
    system RAM and crawl, which looks "stuck" — so the micro-batch is scaled to the card
    and gradient accumulation keeps the *effective* batch at `effective_batch` everywhere.

    Background augmentation: ``mask_dir`` (local splits path) OR ``mask_repo`` (HF
    paired {image, mask, species, split} dataset from
    ``scripts/publish_student_masks_full.py``) enables per-sample Gaussian-blur of
    the background at rate ``bg_aug_p`` — only on the train split.
    """
    eval_tf = eval_tf or eval_transforms()
    train_ds = _make_ds(data, "train", train_tf, mask_dir=mask_dir, mask_repo=mask_repo, bg_aug_p=bg_aug_p)
    val_ds = _make_ds(data, "val", eval_tf)
    test_ds = _make_ds(data, "test", eval_tf)

    dev = env.device
    if dev.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch = 64 if vram > 12 else 32 if vram > 8 else 16 if vram > 6 else 8
    else:
        batch = 16
    num_workers = 0 if os.name == "nt" else 2   # windows spawn hangs -> 0; linux/colab -> parallel
    use_amp = dev.type == "cuda"                 # mixed precision: big speedup on tensor-core GPUs
    accum = max(1, round(effective_batch / batch))

    val = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    test = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    print(f"BATCH {batch} x ACCUM {accum} = eff {batch*accum} | workers {num_workers} | amp {use_amp}")

    mask_key = ""
    if bg_aug_p > 0:
        if mask_repo:
            mask_key = f"hf:{mask_repo}"
        elif mask_dir is not None:
            mask_key = f"local:{Path(mask_dir).name}"
    return Loaders(train_ds, val, test, batch, accum, num_workers, use_amp,
                   train_tf, data.labels, data.n_species,
                   bg_aug_p=bg_aug_p, mask_key=mask_key)
