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
    def __init__(self, df, tf, label2idx):
        self.df = df.reset_index(drop=True)
        self.tf = tf
        self.l = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), self.l[r["species"]]


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


def _make_ds(data: Data, split: str, tf):
    if data._local_splits is not None:
        return _PlantSetLocal(data._local_splits[data._local_splits["split"] == split], tf, data.label2idx)
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


def build_loaders(data: Data, train_tf, env, eval_tf=None, effective_batch: int = 64) -> Loaders:
    """Build val/test loaders + a train dataset, with a VRAM-sized batch.

    Small cards (a 4 GB laptop GPU) don't OOM on Windows — they silently spill to shared
    system RAM and crawl, which looks "stuck" — so the micro-batch is scaled to the card
    and gradient accumulation keeps the *effective* batch at `effective_batch` everywhere.
    """
    eval_tf = eval_tf or eval_transforms()
    train_ds = _make_ds(data, "train", train_tf)
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
