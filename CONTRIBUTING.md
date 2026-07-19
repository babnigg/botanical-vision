# Contributing / development guide

## What this repo is

An **ADSP 32023 (Advanced Computer Vision)** project — a gardener's tool with a CV spine:
fine-grained flowering-plant **classification** (Identify), a **Toolbox** of collected
species, and **Compose**, a garden-design tool. The **computer-vision work is the graded
core** (the models, notebooks, evaluation). The `demo/` app is a separate showcase.

Layout:

```
notebooks/   01–05  data pipeline + training + evaluation   (the CV core)
share/       publish / leaderboard / score — the team model-sharing loop
scripts/     dataset build (download_inaturalist, upload_to_hf) — maintainer-only
demo/        separate showcase app: React frontend + a minimal FastAPI (loads a checkpoint)
```

## Your job: improve models & share them

The work is **making the model better** and sharing it so the team can compare. Two
commands on top of the training notebook:

```bash
# improve the model in notebooks/04_train_improved.ipynb, then share your best:
python -m share.publish --checkpoint checkpoints/resnet50_improved_best.pt --name my-model

# see how everyone's models stack up (same test split, apples-to-apples):
python -m share.leaderboard

# (optional) check your own top-1/top-5 before publishing:
python -m share.score --checkpoint checkpoints/resnet50_improved_best.pt
```

`publish` uploads your weights to the shared **Hugging Face model repo** (not git —
weights never go in git), tagged by your username; `leaderboard` pulls everyone's and
scores them identically. First time only: `huggingface-cli login`.

## Local setup

```bash
cd code/project
python -m venv .venv && .venv\Scripts\activate      # (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Windows without `make`: run the commands directly; otherwise `make help` lists shortcuts.

## The demo app (optional, separate)

A self-contained showcase lives in `demo/` (see `demo/README.md`). Its backend loads a
checkpoint **directly** — no MLflow, no registry. Run its backend + frontend only if you
want to show the classifier as a product; it never affects training or sharing.

## Google Colab

The training/eval notebooks run **unchanged on Colab** — they auto-detect the environment
and load the dataset from Hugging Face when local `../data` isn't present. Mount Drive so
checkpoints survive disconnects (the checkpoint cell handles this). Publishing from Colab
works too: `huggingface-cli login`, then `python -m share.publish ...`.

## Rules for a shared repo (please follow)

- **Keep `README.md` and `.gitignore` current with every change.** This is mandatory.
- Run `make lint` (ruff) before pushing.
- Branch off `main` and open a PR; don't push to `main` directly.
- **Never commit**: model weights (`*.pt`), datasets/images, `.env`, `node_modules/`.
  Data and models both live on Hugging Face (share models with `share.publish`).
- Prioritize the **CV work** — the demo supports the project's grade, it isn't it.
