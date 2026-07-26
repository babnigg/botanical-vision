"""Segmentation training loop, IoU checkpointing, and evaluation.

Mirrors ``bvtrain.train.fit`` but for binary segmentation: BCE + soft-Dice loss,
mean-IoU metric, best-val selection on IoU. Reuses ``checkpoint_dir`` and the
same ``{_last, _best}.pt`` envelope so Colab/Kaggle disconnect recovery works
identically to the classifier pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os

import torch

from .train import _find_resume, checkpoint_dir


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on the plant channel (per-image, averaged over the batch)."""
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


class BceDiceLoss(torch.nn.Module):
    """BCEWithLogits + soft Dice. Combined loss is standard for binary seg —
    BCE stabilizes the pixel-level gradient, Dice keeps things sensible when
    the plant covers only a small fraction of the frame."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * dice_loss(logits, target)


def _iou(logits: torch.Tensor, target: torch.Tensor,
         thresh: float = 0.5, eps: float = 1e-6) -> float:
    """Per-image IoU averaged over the batch (plant class)."""
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * target).sum(dim=(1, 2, 3))
    union = ((preds + target) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def _sig_seg(arch, loaders, optimizer, criterion, epochs, seed) -> str:
    # Hash the criterion's scalar attrs (e.g. BceDiceLoss.bce_weight / dice_weight)
    # too — otherwise tweaking loss weights silently resumes the old checkpoint.
    loss_params = {k: v for k, v in vars(criterion).items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
    cfg = {"model": arch, "epochs": epochs, "batch": loaders.batch, "seed": seed,
           "lrs": [g["lr"] for g in optimizer.param_groups],
           "wd": optimizer.param_groups[0].get("weight_decay", 0),
           "loss": type(criterion).__name__,
           "loss_params": loss_params,
           "aug": str(loaders.train_tf)}
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def _model_out(model, x):
    """Torchvision segmentation models return an OrderedDict; unwrap the main head.

    fit_seg has no aux-head loss term, so silently dropping ``out["aux"]`` would
    waste those gradients — build the model with ``aux_loss=False`` (as notebook
    06 does) or extend the loss before enabling it."""
    out = model(x)
    if isinstance(out, dict):
        assert "aux" not in out, "aux head present but fit_seg has no aux loss; build with aux_loss=False"
        return out["out"]
    return out


def _run_seg_eval(model, dl, dev, criterion, use_amp, desc):
    from tqdm.auto import tqdm
    model.eval()
    tl, iou_sum, n = 0.0, 0.0, 0
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        for x, y in tqdm(dl, desc=desc, leave=False):
            x, y = x.to(dev), y.to(dev)
            out = _model_out(model, x)
            tl += criterion(out, y).item() * len(y)
            iou_sum += _iou(out, y) * len(y)
            n += len(y)
    return tl / n, iou_sum / n


def fit_seg(model, optimizer, loaders, epochs, run_name, env, *,
            scheduler=None, criterion=None, seed=42, ckpt_every=500, fresh=False,
            arch="deeplabv3_mnv3"):
    """Train `model` on binary masks, checkpointing on best val IoU. Returns history dict."""
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    dev = env.device
    criterion = criterion if criterion is not None else BceDiceLoss()
    scaler = torch.amp.GradScaler(enabled=loaders.use_amp)
    accum = loaders.accum

    sig = _sig_seg(arch, loaders, optimizer, criterion, epochs, seed)
    cdir = checkpoint_dir(env)
    last_path = f"{cdir}/{run_name}_last.pt"
    best_path = f"{cdir}/{run_name}_best.pt"

    hist = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    start_epoch, start_step, best_val, resume_run = 0, 0, 0.0, (0.0, 0.0, 0)

    rpath = _find_resume(last_path, run_name, env)
    if rpath and not fresh:
        ck = torch.load(rpath, weights_only=False, map_location=dev)
        if ck["sig"] == sig:
            model.load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            if scheduler is not None and ck.get("sched"):
                scheduler.load_state_dict(ck["sched"])
            start_epoch, start_step, best_val, hist = ck["epoch"], ck["step"], ck["best_val"], ck["hist"]
            resume_run = tuple(ck.get("run", (0.0, 0.0, 0)))
            print(f"resuming: epoch {start_epoch+1}, step {start_step}, best_val_iou {best_val:.3f}")
        else:
            print("existing checkpoint has a different config -> starting fresh")
    else:
        print("fresh run")

    labels = ["background", "plant"]

    def save_ckpt(path, epoch, step, run=(0.0, 0.0, 0)):
        payload = {"sig": sig, "epoch": epoch, "step": step, "run": list(run),
                   "state_dict": model.state_dict(), "opt": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "best_val": best_val, "hist": hist,
                   "labels": labels, "arch": arch}
        if scheduler is not None:
            payload["sched"] = scheduler.state_dict()
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    for ep in range(start_epoch, epochs):
        g = torch.Generator().manual_seed(seed + ep)
        perm = torch.randperm(len(loaders.train_ds), generator=g).tolist()
        skip = start_step if ep == start_epoch else 0
        train_dl = DataLoader(loaders.train_ds, batch_size=loaders.batch,
                              sampler=perm[skip*loaders.batch:], num_workers=loaders.num_workers)
        nbatch = len(train_dl)

        model.train()
        tot, iou_sum, seen = resume_run if (ep == start_epoch and skip > 0) else (0.0, 0.0, 0)
        bar = tqdm(train_dl, desc=f"epoch {ep+1}/{epochs} train", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(bar):
            step = skip + i
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=loaders.use_amp):
                out = _model_out(model, x)
                loss = criterion(out, y)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == nbatch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            tot += loss.item() * len(y)
            iou_sum += _iou(out, y) * len(y)
            seen += len(y)
            bar.set_postfix(loss=f"{tot/seen:.3f}", iou=f"{iou_sum/seen:.3f}")
            if (step + 1) % ckpt_every == 0 and i < nbatch - 1:
                save_ckpt(last_path, ep, step + 1, (tot, iou_sum, seen))
        tl, ti = tot / max(seen, 1), iou_sum / max(seen, 1)

        vl, vi = _run_seg_eval(model, loaders.val, dev, criterion, loaders.use_amp,
                               f"epoch {ep+1}/{epochs} val")
        if scheduler is not None:
            scheduler.step()
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_iou"].append(ti)
        hist["val_iou"].append(vi)
        if vi > best_val:
            best_val = vi
            save_ckpt(best_path, ep + 1, 0)
        save_ckpt(last_path, ep + 1, 0)
        start_step = 0
        print(f"epoch {ep+1}: train_loss {tl:.3f} iou {ti:.3f} | val_loss {vl:.3f} iou {vi:.3f}")

    print(f"best val IoU: {best_val:.3f}")
    return hist


def evaluate_seg(model, loader, env, run_name: str | None = None, thresh: float = 0.5):
    """Mean IoU on a loader. Pass ``run_name`` to score the best-val checkpoint."""
    from tqdm.auto import tqdm
    dev = env.device
    if run_name:
        best = f"{checkpoint_dir(env)}/{run_name}_best.pt"
        if os.path.exists(best):
            ck = torch.load(best, weights_only=False, map_location=dev)
            model.load_state_dict(ck["state_dict"])
    model.eval()
    iou_sum, n = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16,
                                          enabled=(dev.type == "cuda")):
        for x, y in tqdm(loader, desc="test"):
            x, y = x.to(dev), y.to(dev)
            out = _model_out(model, x)
            iou_sum += _iou(out, y, thresh) * len(y)
            n += len(y)
    res = {"iou": iou_sum / n}
    print(f"test mIoU: {res['iou']:.3f}")
    return res


def plot_seg_history(hist):
    """Loss + IoU curves; parallel to ``bvtrain.train.plot_history`` which plots
    loss + accuracy for the classifier."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.set_facecolor("linen")
    axs[0].plot(hist["train_loss"], color="darkorange", label="train")
    axs[0].plot(hist["val_loss"], color="sienna", label="val")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss", weight="bold")
    axs[0].legend()
    axs[1].plot(hist["train_iou"], color="darkorange", label="train")
    axs[1].plot(hist["val_iou"], color="sienna", label="val")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("IoU")
    axs[1].set_title("IoU", weight="bold")
    axs[1].legend()
    plt.tight_layout()
    plt.show()


def load_seg_model(checkpoint_path, env):
    """Rebuild DeepLabV3-MobileNetV3 from a checkpoint. Returns (model, checkpoint)."""
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    ck = torch.load(checkpoint_path, weights_only=False, map_location=env.device)
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=1, aux_loss=False)
    model.load_state_dict(ck["state_dict"])
    model = model.to(env.device).eval()
    return model, ck
