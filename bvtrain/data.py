"""Data loading: species labels, the train/val/test datasets, and dataloaders.

Reads local `../data` files if present (maintainer), else the streamed HF dataset
(Colab / Kaggle / teammates). `build_loaders` sizes the batch to the GPU and picks a
gradient-accumulation factor so the *effective* batch is constant across machines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
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
        import hashlib
        import random as _random

        import numpy as np
        from PIL import Image, ImageFilter
        if self.mask_dir is None or _random.random() >= self.bg_aug_p:
            return img
        mp = self.mask_dir / (hashlib.md5(str(path).encode()).hexdigest() + ".png")
        if not mp.exists():
            return img
        mask = Image.open(mp).convert("L").resize(img.size)
        m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        if m.mean() > 0.98:            # all-plant mask = nothing to blur
            return img
        blurred = img.filter(ImageFilter.GaussianBlur(15))
        out = np.asarray(img) * m + np.asarray(blurred) * (1 - m)
        return Image.fromarray(out.astype("uint8"))

    def __getitem__(self, i):
        import time

        from PIL import Image
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
    def __init__(self, ds, tf, label2idx):
        self.ds = ds
        self.tf = tf
        self.l = label2idx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        return self.tf(ex["image"].convert("RGB")), self.l[ex["species"]]


def _make_ds(data: Data, split: str, tf, mask_dir=None, bg_aug_p=0.0):
    if data._local_splits is not None:
        return _PlantSetLocal(data._local_splits[data._local_splits["split"] == split], tf, data.label2idx,
                              mask_dir=mask_dir if split == "train" else None,
                              bg_aug_p=bg_aug_p if split == "train" else 0.0)
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

    @property
    def n_labels(self) -> int:
        return len(self.labels)


def build_loaders(data: Data, train_tf, env, eval_tf=None, effective_batch: int = 64,
                  mask_dir=None, bg_aug_p: float = 0.0) -> Loaders:
    """Build val/test loaders + a train dataset, with a VRAM-sized batch.

    Small cards (a 4 GB laptop GPU) don't OOM on Windows — they silently spill to shared
    system RAM and crawl, which looks "stuck" — so the micro-batch is scaled to the card
    and gradient accumulation keeps the *effective* batch at `effective_batch` everywhere.
    """
    eval_tf = eval_tf or eval_transforms()
    train_ds = _make_ds(data, "train", train_tf, mask_dir=mask_dir, bg_aug_p=bg_aug_p)
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
    return Loaders(train_ds, val, test, batch, accum, num_workers, use_amp,
                   train_tf, data.labels, data.n_species)
