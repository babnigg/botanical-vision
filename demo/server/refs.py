"""Species reference lookup — a good photo, a short description, and official links.

Pulls a representative photo and a Wikipedia-sourced summary from the iNaturalist
taxa API, and links out to GBIF, iNaturalist, and Wikipedia. Results are cached
in-memory and degrade to links-only if the network or a source is unavailable, so
the endpoint never blocks the UI for long.
"""
from __future__ import annotations

import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import traits

ROOT = Path(__file__).resolve().parents[2]  # demo/server -> demo -> project root
_INAT_MAP: dict | None = None
_CACHE: dict[str, dict] = {}
_HEADERS = {"User-Agent": "BotanicalVision/0.1 (ADSP 32023 course project)"}


def _inat_map() -> dict:
    global _INAT_MAP
    if _INAT_MAP is None:
        try:
            _INAT_MAP = json.loads((ROOT / "data" / "inat_taxon_map.json").read_text(encoding="utf-8"))
        except Exception:
            _INAT_MAP = {}
    return _INAT_MAP


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=6) as r:
        return json.loads(r.read().decode("utf-8"))


def _clean(html: str | None, limit: int = 460) -> str | None:
    if not html:
        return None
    text = re.sub(r"<[^>]+>", "", html).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text or None


def reference(species: str) -> dict:
    if species in _CACHE:
        return _CACHE[species]

    key = traits.species_key(species)
    inat_id = _inat_map().get(species)
    common = image = summary = wiki = None

    try:
        if inat_id:
            data = _get(f"https://api.inaturalist.org/v1/taxa/{inat_id}")
        else:
            data = _get(
                "https://api.inaturalist.org/v1/taxa?rank=species&per_page=1&q="
                + urllib.parse.quote(species)
            )
        results = data.get("results", [])
        if results:
            t = results[0]
            inat_id = inat_id or t.get("id")
            common = t.get("preferred_common_name")
            summary = _clean(t.get("wikipedia_summary"))
            wiki = t.get("wikipedia_url")
            photo = t.get("default_photo") or {}
            image = photo.get("medium_url") or photo.get("url")
    except Exception:
        pass  # links-only fallback below

    links: dict[str, str] = {}
    if key:
        links["gbif"] = f"https://www.gbif.org/species/{key}"
    if inat_id:
        links["inaturalist"] = f"https://www.inaturalist.org/taxa/{inat_id}"
    if wiki:
        links["wikipedia"] = wiki.replace(" ", "_")

    out = {
        "species": species,
        "common": common or traits.taxonomy(species).get("common", ""),
        "image": image,
        "summary": summary,
        "links": links,
    }
    _CACHE[species] = out
    return out


def download_image(url: str) -> bytes | None:
    """Fetch an image's bytes (used to classify a photo given by URL)."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.read()
    except Exception:
        return None


def random_photo() -> dict:
    """A random real plant photo (a random species' iNaturalist image)."""
    names = traits.all_species()
    if not names:
        return {"url": None, "species": None}
    for _ in range(6):
        name = random.choice(names)
        r = reference(name)
        if r.get("image"):
            return {"url": r["image"], "species": name}
    return {"url": None, "species": None}
