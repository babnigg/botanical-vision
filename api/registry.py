"""Model registry view for the app — backed by MLflow, generalized over models.

Each module maps to a registered model name in ``config.MODELS``. A version reads
**live** if it holds the ``production`` alias, **candidate** if registered but not
promoted. Sections without a registered model yet fall back to ``models/planned.json``
(status **planned**). Adding a new section's model = register it in MLflow + add a row
to ``config.MODELS``; nothing in this file changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from mlops import config

ROOT = Path(__file__).resolve().parents[1]
PLANNED_PATH = ROOT / "models" / "planned.json"


def _client():
    config.configure()
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _alias_version(client, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except Exception:
        return None


def _versions(name: str, module: str) -> tuple[list[dict], str | None]:
    """(rows for each registered version, production version or None) for one model."""
    try:
        c = _client()
        prod = _alias_version(c, name, "production")
        staging = _alias_version(c, name, "staging")
        rows = []
        for mv in c.search_model_versions(f"name='{name}'"):
            metric = name
            try:
                acc = c.get_run(mv.run_id).data.metrics.get("val_acc")
                if acc is not None:
                    metric = f"val_acc {acc:.3f}"
            except Exception:
                pass
            rows.append({
                "id": f"{module}-v{mv.version}",
                "name": f"{name} · v{mv.version}",
                "module": module,
                "metric": metric,
                "status": "live" if mv.version == prod else "candidate",
                "version": mv.version,
                "alias": ("production" if mv.version == prod
                          else "staging" if mv.version == staging else None),
            })
        rows.sort(key=lambda m: -int(m["version"]))
        return rows, prod
    except Exception:
        return [], None


def _planned() -> list[dict]:
    try:
        return json.loads(PLANNED_PATH.read_text(encoding="utf-8")).get("models", [])
    except Exception:
        return []


def module_status(module: str) -> str:
    name = config.MODELS.get(module)
    if not name:
        return "prototype"
    _, prod = _versions(name, module)
    return "live" if prod else "prototype"


def summary() -> dict:
    rows: list[dict] = []
    module_live: dict[str, bool] = {}
    for module, name in config.MODELS.items():
        r, prod = _versions(name, module)
        rows += r
        module_live[module] = bool(prod)
    models = rows + _planned()
    return {
        "models": models,
        "modules": {m: ("live" if module_live.get(m) else "prototype")
                    for m in ("identify", "illustrate", "compose")},
        "live_count": sum(1 for m in models if m["status"] == "live"),
        "total": len(models),
    }
