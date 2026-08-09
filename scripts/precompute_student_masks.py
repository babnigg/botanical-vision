"""Precompute plant masks for a training subset with the distilled student
segmenter, for the background-blur augmentation in bvtrain (bg_aug_p).

Masks are written to data/student_masks/<md5-of-image-path>.png (gitignored) —
the key the augmentation looks up. Resumable: existing masks are skipped.

Usage:
    python scripts/precompute_student_masks.py \
        --segmenter checkpoints/deeplabv3_mnv3_seg_best.pt \
        --n-species 100
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

OUT = PROJECT_DIR / "data" / "student_masks"


def load_student(path: str, dev):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=21, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, 1)
    state = ck.get("model", ck.get("state_dict", ck))
    model.load_state_dict(state)
    return model.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", default="checkpoints/deeplabv3_mnv3_seg_best.pt")
    ap.add_argument("--n-species", type=int, default=100)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_student(args.segmenter, dev)

    splits = pd.read_csv(PROJECT_DIR / "data" / "splits.csv")
    keep = sorted(splits["species"].unique())[:args.n_species]
    df = splits[(splits["species"].isin(keep)) & (splits["split"] == args.split)]
    print(f"{len(df)} images ({args.n_species} species, {args.split})")

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1)
    done = skip = 0
    with torch.no_grad():
        for p in tqdm(df["path"], desc="student masks"):
            key = OUT / (hashlib.md5(str(p).encode()).hexdigest() + ".png")
            if key.exists():
                skip += 1
                continue
            img_path = PROJECT_DIR / "data" / str(p).replace("../data/", "")
            try:
                img = Image.open(img_path).convert("RGB")
            except OSError:
                continue
            small = img.resize((224, 224))
            x = torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float().to(dev) / 255
            x = ((x - mean) / std).unsqueeze(0)
            prob = torch.sigmoid(model(x)["out"])[0, 0]
            m = (prob.cpu().numpy() * 255).astype("uint8")
            Image.fromarray(m, mode="L").resize(img.size).save(key)
            done += 1
    print(f"wrote {done}, skipped {skip}")


if __name__ == "__main__":
    main()
