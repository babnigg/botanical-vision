# Botanical Vision — demo app

A **separate, self-contained** showcase for the trained models. It is intentionally
decoupled from the core repo (the notebooks + the `share/` sharing loop): this backend
loads a checkpoint **directly** — no MLflow, no registry, no promotion gate. Deleting or
changing it never touches the model-improvement work.

Modules:
- **Identify** — photo → species (top-5 + taxonomy), served from a checkpoint.
- **Toolbox** — collect identified species; feeds the garden studio.
- **Compose** — a live garden studio: bed size + sun in, planting plan out, drawn as
  an SVG plan view. Served by the masked-diffusion layout model (notebook 11,
  `checkpoints/garden_maskdiff_best.pt`) with species pinning — pick favorites, the
  model infills the rest; falls back to the rule generator without the checkpoint.
  The in-app render is symbolic; styled image renders live in notebook 12.

## Run it (two terminals, from `demo/`)

```bash
# 1. backend  ->  http://localhost:8000
pip install -r server/requirements.txt
uvicorn server.main:app --port 8000

# 2. frontend ->  http://localhost:5173
cd web
npm install
npm run dev
```

Pick the served model on the **Roadmap** tab. It lists your local `checkpoints/*_best.pt`
**and** models published to the shared Hugging Face repo, each with its species count + val
accuracy — you choose, nothing is auto-picked. The choice persists in
`checkpoints/.demo_selected.json`. Identify shows which model is serving and links back to the
Roadmap to change it. With nothing selected (or no PyTorch) Identify returns a labeled demo
stub, so the UI always runs.

## Layout

```
demo/
├── server/   FastAPI — loads a checkpoint directly, serves /api/*
│   ├── main.py · classifier.py · compose.py · models.py · registry.py · refs.py · traits.py · schemas.py
│   └── requirements.txt
└── web/      Vite + React + TS frontend (Identify · Compose · Toolbox · Roadmap)
```
