"""Botanical Vision API.

A thin FastAPI layer over the model registry. Every module endpoint reports
whether it is running a live model or a stub, so the frontend can route the
app up from prototype to functional as checkpoints are deployed.

Run from the project root (code/project):
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api import refs, registry, schemas
from api.ml import classifier, traits

app = FastAPI(title="Botanical Vision API", version="0.1.0")

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


# ---- registry / roadmap (read-only; status is derived from what's deployed) ----
@app.get("/api/registry")
def get_registry() -> dict:
    return registry.summary()


# ---- module 1: identify ----
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


@app.get("/api/species/{name}")
def species(name: str) -> dict:
    tax = traits.SPECIES.get(name)
    if not tax:
        raise HTTPException(404, f"no species '{name}' in the seeded table")
    return {"species": name, **tax}


@app.get("/api/reference/{species}")
def species_reference(species: str) -> dict:
    """A photo + short description + official links for a species."""
    return refs.reference(species)


@app.get("/api/random-photo")
def random_photo() -> dict:
    """A random real plant photo to try the classifier on."""
    return refs.random_photo()


# ---- module 2: illustrate (stub) ----
@app.post("/api/illustrate")
def illustrate(body: schemas.IllustrateRequest) -> dict:
    tax = traits.SPECIES.get(body.species, {})
    fam = tax.get("family", "Asteraceae")
    live = registry.module_status("illustrate") == "live"
    prompt = (f"{body.species}, family {fam} — {body.style.lower()}, "
              f"single specimen, roots to flower, white ground, scientific illustration")
    return {
        "live": live,
        "model": "style-lora" if live else None,
        "species": body.species,
        "style": body.style,
        "prompt": prompt,
        "seed": body.seed or 4471,
        "image": None,  # a real render returns a data URI / URL here
        "note": None if live else "Style LoRA not deployed — prompt is built, render is a preview stub.",
    }


# ---- module 3: compose ----
@app.get("/api/zone/{zip_code}")
def zone(zip_code: str) -> dict:
    return {"zip": zip_code, "zone": traits.zone_from_zip(zip_code)}


@app.post("/api/arrange")
def arrange(body: schemas.ArrangeRequest) -> dict:
    plan = traits.arrange(body.aspect, body.zone, body.area, body.toolbox)
    plan["live"] = registry.module_status("compose") == "live"
    plan["note"] = None if plan["live"] else "Arrangement runs on the seeded trait table; the full engine is a Module-3 objective."
    return plan


@app.post("/api/render")
def render(body: schemas.RenderRequest) -> dict:
    live = registry.module_status("compose") == "live"
    return {
        "live": live,
        "model": "landscape-cn" if live else None,
        "style": body.style,
        "as_plate": body.as_plate,
        "image": None,  # ControlNet-seg render returns an image here
        "note": None if live else "Render model not deployed — showing a vector preview stub.",
    }
