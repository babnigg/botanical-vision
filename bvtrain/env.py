"""Environment detection: device + which platform we're on (local / Colab / Kaggle)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Env:
    device: torch.device
    on_colab: bool
    on_kaggle: bool
    use_local: bool
    hf_repo: str = "dbabnigg/botanical-vision-256"
    hf_masks_repo: str | None = None    # companion masks dataset for notebook 07

    @property
    def kind(self) -> str:
        return "colab" if self.on_colab else "kaggle" if self.on_kaggle else "local"


def setup(hf_repo: str = "dbabnigg/botanical-vision-256",
          hf_masks_repo: str | None = None,
          data_path: str = "../data/splits.csv") -> Env:
    """Detect device + platform. Uses local data if present, else the HF dataset."""
    # kaggle first: its image also ships google.colab, so import-based colab
    # detection false-positives there (checkpoints would silently go to /content)
    on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    on_colab = not on_kaggle and os.environ.get("COLAB_RELEASE_TAG") is not None
    if not on_kaggle and not on_colab:
        try:
            import google.colab  # noqa: F401
            on_colab = True
        except ImportError:
            pass
    use_local = Path(data_path).exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env(device, on_colab, on_kaggle, use_local, hf_repo, hf_masks_repo)
    print(f"device: {device} | data: {'local' if use_local else 'huggingface'} | env: {env.kind}")
    return env
