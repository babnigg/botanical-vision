# Contributing / development guide

## What this repo is

An **ADSP 32023 (Advanced Computer Vision)** project: fine-grained flowering-plant
classification, botanical-illustration generation, and sketch-to-landscape design. The
**computer-vision work is the graded core** — the models, the notebooks, and their
evaluation. The `api/` + `app/` product and the `mlops/` model-registry machinery are
supporting **best-practice infrastructure** (informed by ADSP 32021 MLOps); keep them
working, but they exist to serve the CV work, not the other way round.

Layout:

```
notebooks/   01–05  data pipeline + training + evaluation   (the CV core)
mlops/       MLflow packaging, registry, evaluation gate, Prefect flow
api/         FastAPI — serves the @production model from MLflow
app/         Vite + React + TS frontend
models/      README.md (registry flow) + planned.json (untrained-model roadmap rows)
```

## Local setup (laptop)

```bash
cd code/project
python -m venv .venv && .venv\Scripts\activate      # (macOS/Linux: source .venv/bin/activate)
pip install -r api/requirements.txt                 # includes mlflow + torch
cd app && npm install && cd ..
```

Run (three terminals, from `code/project`):

```bash
python -m uvicorn api.main:app --port 8000          # API  → localhost:8000
cd app && npm run dev                               # UI   → localhost:5173
python -m mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db --port 5000   # registry UI
```

Windows without `make`: run the commands directly; otherwise `make help` lists shortcuts.

## The model lifecycle (how a model reaches the app)

1. **Train** in `notebooks/03_train_classifier.ipynb` / `04_train_improved.ipynb`.
2. **Register** → a versioned pyfunc model in MLflow:
   `python -m mlops.register_checkpoint --checkpoint checkpoints/<ckpt>.pt`
3. **Evaluate** on the streamed HF test split:
   `python -m mlops.evaluate --version <v> --limit 300`
4. **Promote** through the champion/challenger gate (sets `@production` only if it beats
   the current champion): `python -m mlops.promote --challenger <v> --margin 0.005`
5. The **API serves** `models:/botanical-classifier@production` automatically.

`python -m mlops.flows.train_flow --checkpoint <ckpt>` runs register → evaluate → gate as
one Prefect flow (`pip install prefect` first).

## Extending to another section's model (illustrate / compose)

The registry is generalized, so wiring a new model is mostly data + a small flavor:

1. Train the model (e.g. the style LoRA, the ControlNet render model).
2. Write a **pyfunc flavor** for it (its own `predict`/inputs/outputs) — the classifier's
   `mlops/model.py` is classifier-specific; don't reuse it verbatim.
3. Register it under a new name (e.g. `botanical-style-lora`).
4. Add a row to `mlops/config.py` → `MODELS` (`"illustrate": "botanical-style-lora"`). The
   roadmap + module status pick it up with no other change.
5. Give it a **model-appropriate eval/gate** — generators use FID/CLIP, not top-1, so add a
   generator-specific evaluate/promote rather than reusing the classifier's.
6. Remove its row from `models/planned.json`; update `README.md` + `models/README.md`.

Known sharp edges (so you don't hit them blind):
- Serving a generator is **GPU/heavy** — it belongs in its own endpoint/service, not the
  current CPU classifier API.
- The classifier `evaluate.py` computes top-1/top-5; **don't** point it at a generator.

## Google Colab (train + register there)

Colab works for the CV notebooks unchanged, and for registering models — but Colab disk is
wiped on disconnect, so the MLflow store must live somewhere persistent.

```python
!git clone https://github.com/babnigg/botanical-vision.git && cd botanical-vision
!pip install -q -r api/requirements.txt

# Option A — personal, quick: put the registry on Google Drive
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////content/drive/MyDrive/botanical-vision/mlflow.db"

# ...train (notebooks 03/04), then:
!python -m mlops.register_checkpoint --checkpoint checkpoints/resnet50_baseline.pt --alias production
!python -m mlops.promote --challenger 2 --margin 0.005 --limit 300
```

Serving the app on Colab isn't the intent (Colab isn't a server host); for a quick demo you
can tunnel uvicorn with cloudflared/ngrok as in the 32021 lab, but serving belongs on a
laptop or a Hugging Face Space. **The frontend/app is not needed to train or register.**

## Shared team registry (recommended once more than one person deploys)

A local SQLite/Drive registry is **per-person** — teammates don't see each other's versions.
For a shared registry, point MLflow at **Databricks Managed MLflow** (the 32021 lab tool) —
one env var, no code change:

```python
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["DATABRICKS_HOST"] = "https://<workspace>.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "<token>"
```

Then everyone (Colab or laptop) registers/reads the same versions.

## Rules for a shared repo (please follow)

- **Keep `README.md` and `.gitignore` current with every change.** New artifact dir → ignore
  it; changed run steps → update the README. This is mandatory, not optional.
- Run `make lint` (ruff) and `make test` (pytest) before pushing; **CI must pass**
  (`.github/workflows/ci.yml` runs ruff + pytest + docker build).
- Branch off `main` and open a PR; don't push to `main` directly.
- **Never commit**: model weights (`*.pt`), datasets/images, `mlflow.db`, `mlruns/`,
  `.env`, `node_modules/`. Data lives on Hugging Face; weights in the registry.
- Prioritize the **CV work** — deployment infra supports the project's grade, it isn't it.
