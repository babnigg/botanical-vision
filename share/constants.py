"""Constants shared across the model-sharing loop.

The eval transform spec (resize/crop/normalize) and the architecture travel inside
every published bundle, but these are the source-of-truth defaults used to rebuild a
model from a bundle and to keep publish + leaderboard in agreement.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # code/project

ARCH = "resnet50"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TOP_K = 5

# The downscaled dataset the notebooks train on; its test split is the common
# yardstick the leaderboard ranks every shared model against.
HF_DATA_REPO = "dbabnigg/botanical-vision-256"
