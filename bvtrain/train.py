"""The training loop, evaluation, and resumable checkpointing.

`fit` runs a mixed-precision, gradient-accumulating loop that checkpoints every N steps
and resumes exactly where it left off — surviving a Colab/Kaggle disconnect. Checkpoints
go somewhere that persists: Colab -> mounted Drive, Kaggle -> /kaggle/working (saved as
the kernel's output), local -> ../checkpoints.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import torch


def checkpoint_dir(env) -> str:
    """Where checkpoints live for this platform (created if missing)."""
    if env.on_colab:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            print("run the 'Colab: Mount Google Drive to Server' command, then rerun this cell")
        d = "/content/drive/MyDrive/botanical-vision/checkpoints"
    elif env.on_kaggle:
        d = "/kaggle/working"
    else:
        d = "../checkpoints"
    os.makedirs(d, exist_ok=True)
    return d


def _find_resume(last_path: str, run_name: str, env) -> str | None:
    if os.path.exists(last_path):
        return last_path
    if env.on_kaggle:  # a previous run's checkpoint arrives as an attached input dataset
        hits = sorted(glob.glob(f"/kaggle/input/**/{run_name}_last.pt", recursive=True))
        if hits:
            return hits[-1]
    return None


def _signature(arch, loaders, optimizer, criterion, epochs, seed) -> str:
    # a run only resumes a checkpoint with the SAME setup (else it's a different experiment)
    cfg = {"model": arch, "n_species": loaders.n_species, "epochs": epochs,
           "n_labels": loaders.n_labels, "batch": loaders.batch, "seed": seed,
           "lrs": [g["lr"] for g in optimizer.param_groups],
           "wd": optimizer.param_groups[0].get("weight_decay", 0),
           "label_smoothing": getattr(criterion, "label_smoothing", 0.0),
           "aug": str(loaders.train_tf)}
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def _run_eval(model, dl, dev, criterion, use_amp, desc):
    from tqdm.auto import tqdm
    model.eval()
    tl, cor, n = 0.0, 0, 0
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        for x, y in tqdm(dl, desc=desc, leave=False):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            tl += criterion(out, y).item() * len(y)
            cor += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tl / n, cor / n


def fit(model, optimizer, loaders, epochs, run_name, env, *,
        scheduler=None, criterion=None, seed=42, ckpt_every=1000, fresh=False, arch="resnet50"):
    """Train `model`, checkpointing + resuming automatically. Returns the history dict."""
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    dev = env.device
    criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=loaders.use_amp)
    accum = loaders.accum

    sig = _signature(arch, loaders, optimizer, criterion, epochs, seed)
    cdir = checkpoint_dir(env)
    last_path = f"{cdir}/{run_name}_last.pt"
    best_path = f"{cdir}/{run_name}_best.pt"

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_epoch, start_step, best_val, resume_run = 0, 0, 0.0, (0.0, 0, 0)

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
            resume_run = tuple(ck.get("run", (0.0, 0, 0)))
            print(f"resuming: epoch {start_epoch+1}, step {start_step}, best_val {best_val:.3f}")
        else:
            print("existing checkpoint has a different config -> starting fresh")
    else:
        print("fresh run")

    def save_ckpt(path, epoch, step, run=(0.0, 0, 0)):
        # (epoch, step) = the NEXT batch to run. atomic write survives a mid-save disconnect.
        payload = {"sig": sig, "epoch": epoch, "step": step, "run": list(run),
                   "state_dict": model.state_dict(), "opt": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "best_val": best_val, "hist": hist,
                   "labels": loaders.labels}
        if scheduler is not None:
            payload["sched"] = scheduler.state_dict()
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    for ep in range(start_epoch, epochs):
        # deterministic per-epoch order so a mid-epoch resume continues exactly where it stopped
        g = torch.Generator().manual_seed(seed + ep)
        perm = torch.randperm(len(loaders.train_ds), generator=g).tolist()
        skip = start_step if ep == start_epoch else 0
        train_dl = DataLoader(loaders.train_ds, batch_size=loaders.batch,
                              sampler=perm[skip*loaders.batch:], num_workers=loaders.num_workers)
        nbatch = len(train_dl)

        model.train()
        tot, cor, seen = resume_run if (ep == start_epoch and skip > 0) else (0.0, 0, 0)
        bar = tqdm(train_dl, desc=f"epoch {ep+1}/{epochs} train", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(bar):
            step = skip + i
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=loaders.use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss / accum).backward()          # accumulate; effective batch = BATCH*ACCUM
            if (i + 1) % accum == 0 or (i + 1) == nbatch:   # step once per ACCUM microbatches (and last)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            tot += loss.item() * len(y)
            cor += (out.argmax(1) == y).sum().item()
            seen += len(y)
            bar.set_postfix(loss=f"{tot/seen:.3f}", acc=f"{cor/seen:.3f}")
            if (step + 1) % ckpt_every == 0 and i < nbatch - 1:
                save_ckpt(last_path, ep, step + 1, (tot, cor, seen))
        tl, ta = tot / max(seen, 1), cor / max(seen, 1)

        vl, va = _run_eval(model, loaders.val, dev, criterion, loaders.use_amp, f"epoch {ep+1}/{epochs} val")
        if scheduler is not None:
            scheduler.step()
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        if va > best_val:
            best_val = va
            save_ckpt(best_path, ep + 1, 0)
        save_ckpt(last_path, ep + 1, 0)   # epoch boundary
        start_step = 0
        print(f"epoch {ep+1}: train_loss {tl:.3f} acc {ta:.3f} | val_loss {vl:.3f} acc {va:.3f}")

    print(f"best val acc: {best_val:.3f}")
    return hist


def evaluate(model, loader, env, topk=(1, 5), run_name: str | None = None):
    """Top-k accuracy on a loader. Pass `run_name` to score the best-val checkpoint."""
    from tqdm.auto import tqdm
    dev = env.device
    if run_name:
        best = f"{checkpoint_dir(env)}/{run_name}_best.pt"
        if os.path.exists(best):
            ck = torch.load(best, weights_only=False, map_location=dev)
            model.load_state_dict(ck["state_dict"])
    model.eval()
    kmax = max(topk)
    hits = {k: 0 for k in topk}
    n = 0
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
        for x, y in tqdm(loader, desc="test"):
            x, y = x.to(dev), y.to(dev)
            pred = model(x).topk(kmax, 1).indices
            for k in topk:
                hits[k] += (pred[:, :k] == y.unsqueeze(1)).any(1).sum().item()
            n += len(y)
    res = {f"top{k}": hits[k] / n for k in topk}
    for k in topk:
        print(f"test top-{k}: {res[f'top{k}']:.3f}")
    return res


def load_model(checkpoint_path, env):
    """Rebuild ResNet-50 from a checkpoint's labels + weights. Returns (model, checkpoint)."""
    from torch import nn
    from torchvision import models
    ck = torch.load(checkpoint_path, weights_only=False, map_location=env.device)
    labels = ck["labels"]
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(ck["state_dict"])
    model = model.to(env.device).eval()
    return model, ck


def plot_history(hist):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.set_facecolor("linen")
    axs[0].plot(hist["train_loss"], color="darkorange", label="train")
    axs[0].plot(hist["val_loss"], color="sienna", label="val")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss", weight="bold")
    axs[0].legend()
    axs[1].plot(hist["train_acc"], color="darkorange", label="train")
    axs[1].plot(hist["val_acc"], color="sienna", label="val")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy", weight="bold")
    axs[1].legend()
    plt.tight_layout()
    plt.show()
