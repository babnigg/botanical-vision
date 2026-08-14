"""Measure each palette species' real bloom and foliage colors from dataset photos.

Writes bvtrain/species_colors.json (committed) so the demo's render painter has
grounded colors everywhere, even without the local photo archive. Peak-hue-bin
extraction over saturated non-green non-soil pixels, a white-flower path, and a
curator veto (the measured color must roughly agree in hue with the curated
palette color, or the palette color wins).

    python scripts/measure_species_colors.py
"""
from __future__ import annotations

import colorsys
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from bvtrain import garden as g  # noqa: E402

STANDIN = {"Nepeta racemosa": "Nepeta cataria",
           "Calamagrostis acutiflora": "Calamagrostis arundinacea",
           "Sedum telephium": "Hylotelephium telephium"}


def measure_colors():
    df = pd.read_csv(PROJECT_DIR / "data" / "splits.csv", usecols=["species", "path", "split"])
    df = df[df["split"] == "train"]
    rng = random.Random(0)
    out = {}
    for p in g.PALETTE:
        key = STANDIN.get(p["name"], p["name"])
        paths = df[df["species"] == key]["path"].tolist()
        px, pale_n, sat_n, leaves = [], 0, 0, []
        for pth in rng.sample(paths, min(10, len(paths))):
            try:
                im = Image.open(PROJECT_DIR / "data" / pth.replace("../data/", "")).convert("RGB").resize((128, 128))
            except OSError:
                continue
            a = np.asarray(im).astype(np.float32) / 255
            hsv = np.asarray(im.convert("HSV")).astype(np.float32)
            h, s, v = hsv[..., 0] / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
            green = (h > 0.17) & (h < 0.45) & (s > 0.15) & (v > 0.1)
            soil = (h > 0.02) & (h < 0.16) & (v < 0.55)
            bloom = (~((h > 0.15) & (h < 0.47))) & (s > 0.4) & (v > 0.35) & (~soil)
            pale = (s < 0.18) & (v > 0.8)
            pale_n += int(pale.sum())
            sat_n += int(bloom.sum())
            if bloom.any():
                px.append(np.stack([h[bloom], a[bloom][:, 0], a[bloom][:, 1], a[bloom][:, 2]], 1))
            if green.mean() > 0.05:
                leaves.append(a[green])
        if px:
            allpx = np.concatenate(px)
            bins = np.floor(allpx[:, 0] * 18).astype(int) % 18
            peak = np.bincount(bins, minlength=18).argmax()
            inpeak = np.isin(bins, [(peak - 1) % 18, peak, (peak + 1) % 18])
            bc = np.median(allpx[inpeak, 1:], 0)
        else:
            bc = np.array(to_rgb(p["color"]))
        if pale_n > 3 * max(sat_n, 1):
            bc = np.array([0.93, 0.92, 0.86])
        mh = colorsys.rgb_to_hsv(*bc)[0]
        ph, ps, _ = colorsys.rgb_to_hsv(*to_rgb(p["color"]))
        dh = min(abs(mh - ph), 1 - abs(mh - ph))
        mx, mn = float(bc.max()), float(bc.min())
        muddy = mx < 0.45 or ((mx - mn) < 0.12 and mx < 0.8)
        if ps < 0.15:
            bc = np.array([0.93, 0.92, 0.86])
        elif muddy or dh > 0.17:
            bc = np.array(to_rgb(p["color"]))
        fc = np.median(np.concatenate(leaves), 0) if leaves else np.array([0.30, 0.42, 0.22])
        out[p["name"]] = {"bloom": [round(float(x), 3) for x in bc],
                          "leaf": [round(float(x), 3) for x in fc],
                          "measured_bloom": bool(px)}
    return out


if __name__ == "__main__":
    colors = measure_colors()
    dst = PROJECT_DIR / "bvtrain" / "species_colors.json"
    json.dump(colors, open(dst, "w"), indent=1)
    print("wrote", dst)
