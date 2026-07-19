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

    @property
    def kind(self) -> str:
        return "colab" if self.on_colab else "kaggle" if self.on_kaggle else "local"


def setup(hf_repo: str = "dbabnigg/botanical-vision-256",
          data_path: str = "../data/splits.csv") -> Env:
    """Detect device + platform. Uses local data if present, else the HF dataset."""
    try:
        import google.colab  # noqa: F401
        on_colab = True
    except ImportError:
        on_colab = False
    on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    use_local = Path(data_path).exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env(device, on_colab, on_kaggle, use_local, hf_repo)
    print(f"device: {device} | data: {'local' if use_local else 'huggingface'} | env: {env.kind}")
    return env
