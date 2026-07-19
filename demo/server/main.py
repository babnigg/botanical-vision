"""Botanical Vision demo API — a thin FastAPI layer that loads a checkpoint directly.

Serves the React frontend in ../web. Identify runs the trained classifier (or a stub
if no checkpoint/torch); Compose is a placeholder garden-design tool backed by the
seeded trait table. Run from the ``demo/`` folder:

    uvicorn server.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import classifier, models, refs, registry, schemas, traits

app = FastAPI(title="Botanical Vision Demo", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "torch": classifier.torch_available(),
        "model": classifier.served_model(),
    }


@app.get("/api/registry")
def get_registry() -> dict:
    return registry.summary()


# ---- model picker: list local + shared models, choose which one Identify serves ----
@app.get("/api/models")
def get_models() -> dict:
    # cheap: metadata only (cached), no model weights loaded just to list.
    aid = classifier.active_id()
    return {"models": models.list_all(aid), "active_id": aid}


@app.post("/api/models/select")
def select_model(body: schemas.SelectRequest) -> dict:
    # this is where the (possibly slow) download + load happens — the frontend shows a
    # "loading/downloading" state until it returns. On failure the selection is cleared.
    classifier.set_active(body.id)
    active = classifier.served_model()   # forces the load/download now
    if active is None:
        classifier.clear_active()
        return {"ok": False, "active_id": None}
    return {"ok": True, "active_id": active["id"]}


# ---- Identify ----
@app.post("/api/classify")
async def classify(file: UploadFile | None = File(default=None),
                   url: str | None = Form(default=None),
                   sample: str | None = Form(default=None)) -> dict:
    image_bytes = await file.read() if file is not None else None
    if image_bytes is None and url:
        image_bytes = refs.download_image(url)
    result = classifier.predict(image_bytes, sample)
    result["module_status"] = registry.module_status("identify")
    return result


@app.get("/api/reference/{species}")
def species_reference(species: str) -> dict:
    """A photo + short description + official links for a species."""
    return refs.reference(species)


@app.get("/api/random-photo")
def random_photo() -> dict:
    """A random real plant photo to try the classifier on."""
    return refs.random_photo()


# ---- Compose (garden design — placeholder engine over the seeded trait table) ----
@app.get("/api/zone/{zip_code}")
def zone(zip_code: str) -> dict:
    return {"zip": zip_code, "zone": traits.zone_from_zip(zip_code)}


@app.post("/api/arrange")
def arrange(body: schemas.ArrangeRequest) -> dict:
    plan = traits.arrange(body.aspect, body.zone, body.area, body.toolbox)
    plan["live"] = False
    plan["note"] = ("Arrangement runs on the seeded trait table; the full engine is a "
                    "Compose (garden-design) objective.")
    return plan
