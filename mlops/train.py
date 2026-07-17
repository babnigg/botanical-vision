"""Training entrypoint for the pipeline.

Full, resumable training lives in ``notebooks/03_train_classifier.ipynb`` and
``04_train_improved.ipynb`` (they auto-detect laptop vs Colab and stream the HF
``-256`` dataset). For the orchestrated pipeline we expose a thin ``train()`` that
returns a checkpoint path: pass an already-trained checkpoint (the common "promote
what we trained" path), or extend this to import the notebooks' training loop.
"""
from __future__ import annotations

from pathlib import Path

from mlops import config


def train(checkpoint: str | None = None, n_species: int | None = None,
          epochs: int | None = None) -> str:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = config.ROOT / checkpoint
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return str(p)
    raise NotImplementedError(
        "Automated retraining isn't wired for the Core tier. Train in notebooks 03/04 "
        "(or extract their loop into this function), then pass a --checkpoint to the flow."
    )
