"""Botanical Vision demo backend — a minimal, self-contained inference server.

Separate from the core repo (notebooks + share/): it loads a trained checkpoint
directly (no MLflow, no registry) and serves the React frontend in ../web. Run it
from the ``demo/`` folder:

    uvicorn server.main:app --port 8000
"""
