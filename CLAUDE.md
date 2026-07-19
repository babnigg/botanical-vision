# Repo guide for AI assistants (and humans in a hurry)

**Project:** ADSP 32023 (Advanced Computer Vision). A gardener's tool with a
computer-vision spine: **Identify** a flowering plant from a photo → collect it in a
**Toolbox** → **Compose** a garden design. The graded core is the CV models in
`notebooks/`; the `demo/` app is a separate showcase.

**This repo was deliberately kept small** (the MLflow registry, FastAPI service, Docker,
and CI were removed). The hard rules:

- **The team loop is two commands** — the point of the repo for the class:
  `python -m share.publish --checkpoint <ckpt> --name <name>` shares a trained model to the
  shared HF model repo; `python -m share.leaderboard` ranks everyone's on the same test
  split. `python -m share.score --checkpoint <ckpt>` checks your own first. No MLflow, no
  registry, no server.
- **`share/` is lean and model-agnostic** — don't reintroduce a registry, promotion gate,
  Docker, or CI. Weights travel via the HF model repo, never git.
- **`demo/` is a separate, self-contained app** (React + a minimal FastAPI that loads a
  checkpoint directly). Keep it decoupled — never make the notebooks or `share/` depend on it.
- **Notebooks are the graded work.** 01/02 are maintainer-only (don't re-run); teammates
  run 03 → 05. Model experiments go in `04_train_improved` (leave it be if it's mid-run).
- **Notebook plumbing lives in `bvtrain/`** (env detection, data loading, resumable
  checkpointing across local/Colab/Kaggle, the training loop, eval). Keep the *model /
  augmentation / optimizer* in the notebook; put reusable machinery in `bvtrain/`. The
  notebooks bootstrap it by cloning the repo on Colab/Kaggle. Kaggle headless-run tooling
  is in `kaggle/` (see its README).
- **Constants must match the notebooks:** ImageNet normalization
  (`MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`), `IMG_SIZE=224`, top-5 — the
  transfer-learning (Week 4) standard. `share/constants.py` mirrors notebooks 03/04/05.
- Keep `README.md` and `.gitignore` current with pipeline changes. Run `make lint` before
  proposing a commit.
