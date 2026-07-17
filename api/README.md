# Botanical Vision — API (FastAPI)

A thin Python service over the **MLflow Model Registry**. Every endpoint reports whether
it is serving a **live model** or a **stub**, so the frontend routes the app up from
prototype to functional as models are promoted. The API never imports training code — it
reads the registry (`sqlite:///mlops/mlflow.db` by default) and serves the model carrying
the `production` alias.

## Run (from `code/project/`)

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS/Linux:           source .venv/bin/activate

pip install -r api/requirements.txt
python -m uvicorn api.main:app --port 8000    # http://localhost:8000
```

Use `python -m uvicorn` (not bare `uvicorn`) — on Windows the console script often
isn't on PATH when uvicorn is installed into the user site-packages. For live
reload, `python -m pip install --user watchfiles` first, then add `--reload`.

Open `http://localhost:8000/docs` for the interactive API.

## Live inference vs. stubs

The API runs and serves believable stubs **without torch installed** — enough to
develop the whole frontend offline. To serve real classifier predictions:

```bash
pip install torch torchvision            # CUDA build per the project README for GPU
```

Then register a checkpoint into MLflow so the API can serve it:

```bash
python -m mlops.register_checkpoint --checkpoint checkpoints/resnet50_baseline.pt --alias production
```

The API loads `models:/botanical-classifier@production` — a packaged pyfunc that carries
its own transform + labels. If no production model exists, or torch/mlflow can't load it,
it falls back to a labeled stub. See the top-level `models/README.md` for the full
register → evaluate → gate → serve flow.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/api/health` | liveness |
| GET  | `/api/registry` | models + per-module status + live count |
| POST | `/api/registry/{id}/status` | route a model up (`planned`→`training`→`live`) |
| POST | `/api/classify` | Module 1 — top-5 species (multipart `file` or form `sample`) |
| GET  | `/api/species/{name}` | taxonomy for a species |
| POST | `/api/illustrate` | Module 2 — builds the prompt (render stub until LoRA deployed) |
| GET  | `/api/zone/{zip}` | zip → USDA hardiness zone (stub) |
| POST | `/api/arrange` | Module 3 — fill a bed from the seeded trait table |
| POST | `/api/render` | Module 3 — plan render (stub until ControlNet deployed) |

## Layout

```
api/
  main.py          # routes
  registry.py      # queries MLflow for versions + per-module status
  schemas.py       # request models
  ml/
    classifier.py  # loads @production pyfunc from MLflow, predict + graceful stub
    traits.py      # taxonomy + seeded garden trait table + arrange()
```
