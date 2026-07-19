"""Static roadmap view for the demo — no MLflow.

Identify is **live** once you've selected a model to serve (see /api/models); Compose is a
planned placeholder (its trait table + arrangement engine are the garden-design objective).
Planned rows come from `models/planned.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import classifier

ROOT = Path(__file__).resolve().parents[2]
PLANNED_PATH = ROOT / "models" / "planned.json"

MODULES = ("identify", "compose")


def _planned() -> list[dict]:
    try:
        return json.loads(PLANNED_PATH.read_text(encoding="utf-8")).get("models", [])
    except Exception:
        return []


def module_status(module: str) -> str:
    if module == "identify":
        return "live" if classifier.active_id() else "prototype"
    return "prototype"


def summary() -> dict:
    models: list[dict] = []
    aid = classifier.active_id()
    if aid:
        models.append({
            "id": "identify-active",
            "name": f"serving: {aid.split(':', 1)[1]}",
            "module": "identify",
            "metric": aid.split(':', 1)[0],   # "local" | "shared"
            "status": "live",
        })
    models += _planned()
    return {
        "models": models,
        "modules": {m: ("live" if module_status(m) == "live" else "prototype") for m in MODULES},
        "live_count": sum(1 for m in models if m["status"] == "live"),
        "total": len(models),
    }
