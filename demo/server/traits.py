"""Taxonomy + a small curated garden trait table, with the arrangement stub.

The real trait table is a Compose (garden-design) objective (USDA PLANTS + Perenual
+ Trefle + Wikidata, joined on a WFO-normalized binomial). This is a hand-seeded
subset so the arrangement engine has something real to reason over today.
"""
from __future__ import annotations

import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # demo/server -> demo -> project root

# species -> taxonomy (a slice of the 4,094-label vocabulary)
SPECIES: dict[str, dict] = {
    "Echinacea purpurea": {"common": "eastern purple coneflower", "genus": "Echinacea", "family": "Asteraceae", "order": "Asterales", "iucn": "LC"},
    "Echinacea pallida": {"common": "pale purple coneflower", "genus": "Echinacea", "family": "Asteraceae", "order": "Asterales", "iucn": "LC"},
    "Echinacea angustifolia": {"common": "narrow-leaf coneflower", "genus": "Echinacea", "family": "Asteraceae", "order": "Asterales", "iucn": "LC"},
    "Rudbeckia hirta": {"common": "black-eyed Susan", "genus": "Rudbeckia", "family": "Asteraceae", "order": "Asterales", "iucn": "LC"},
    "Ratibida pinnata": {"common": "grey-head coneflower", "genus": "Ratibida", "family": "Asteraceae", "order": "Asterales", "iucn": "LC"},
    "Iris versicolor": {"common": "northern blue flag", "genus": "Iris", "family": "Iridaceae", "order": "Asparagales", "iucn": "LC"},
    "Iris virginica": {"common": "Virginia iris", "genus": "Iris", "family": "Iridaceae", "order": "Asparagales", "iucn": "LC"},
    "Iris setosa": {"common": "beachhead iris", "genus": "Iris", "family": "Iridaceae", "order": "Asparagales", "iucn": "LC"},
    "Iris lacustris": {"common": "dwarf lake iris", "genus": "Iris", "family": "Iridaceae", "order": "Asparagales", "iucn": "NT"},
    "Sisyrinchium montanum": {"common": "blue-eyed grass", "genus": "Sisyrinchium", "family": "Iridaceae", "order": "Asparagales", "iucn": "LC"},
}

# garden traits — bloom as [start_month, end_month]; 0,0 == foliage only
GARDEN: list[dict] = [
    {"sp": "Baptisia australis", "sun": "sun", "height": 110, "bloom": [5, 6], "color": "indigo", "role": "structural", "note": "N-fixer backbone"},
    {"sp": "Amsonia hubrichtii", "sun": "sun", "height": 90, "bloom": [5, 5], "color": "blue", "role": "structural", "note": "gold fall foliage"},
    {"sp": "Echinacea purpurea", "sun": "sun", "height": 90, "bloom": [6, 8], "color": "purple", "role": "seasonal", "note": "midsummer anchor"},
    {"sp": "Rudbeckia hirta", "sun": "sun", "height": 60, "bloom": [7, 9], "color": "yellow", "role": "seasonal", "note": "long yellow run"},
    {"sp": "Salvia nemorosa", "sun": "sun", "height": 45, "bloom": [5, 7], "color": "violet", "role": "seasonal", "note": "early spikes, rebloomer"},
    {"sp": "Symphyotrichum novae-angliae", "sun": "sun", "height": 120, "bloom": [9, 10], "color": "purple", "role": "seasonal", "note": "fills the fall gap"},
    {"sp": "Solidago speciosa", "sun": "sun", "height": 90, "bloom": [8, 9], "color": "yellow", "role": "seasonal", "note": "pollinator magnet"},
    {"sp": "Sporobolus heterolepis", "sun": "sun", "height": 60, "bloom": [8, 9], "color": "straw", "role": "groundcover", "note": "matrix grass"},
    {"sp": "Schizachyrium scoparium", "sun": "sun", "height": 90, "bloom": [8, 9], "color": "russet", "role": "structural", "note": "upright winter structure"},
    {"sp": "Nepeta racemosa", "sun": "sun", "height": 45, "bloom": [5, 9], "color": "blue", "role": "groundcover", "note": "edge softener, long bloom"},
    {"sp": "Geranium Rozanne", "sun": "part", "height": 45, "bloom": [6, 9], "color": "blue", "role": "groundcover", "note": "weaves between clumps"},
    {"sp": "Hosta Halcyon", "sun": "shade", "height": 45, "bloom": [7, 7], "color": "lilac", "role": "structural", "note": "shade backbone"},
    {"sp": "Tiarella cordifolia", "sun": "shade", "height": 25, "bloom": [4, 5], "color": "white", "role": "groundcover", "note": "spring shade carpet"},
    {"sp": "Heuchera villosa", "sun": "shade", "height": 40, "bloom": [7, 8], "color": "cream", "role": "groundcover", "note": "evergreen foliage"},
    {"sp": "Dryopteris marginalis", "sun": "shade", "height": 60, "bloom": [0, 0], "color": "green", "role": "structural", "note": "texture, no bloom"},
]
MON = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

_CSV_TAX: dict[str, dict] | None = None


def _csv_taxonomy() -> dict[str, dict]:
    """species -> {genus, family, order, iucn} from data/selected_species.csv."""
    global _CSV_TAX
    if _CSV_TAX is None:
        _CSV_TAX = {}
        path = _ROOT / "data" / "selected_species.csv"
        try:
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    _CSV_TAX[row["species"]] = {
                        "genus": row.get("genus", ""),
                        "family": row.get("family", ""),
                        "order": row.get("order", ""),
                        "iucn": (row.get("iucnRedListCategory") or "").strip() or "LC",
                        "key": row.get("speciesKey", ""),
                    }
        except Exception:
            pass
    return _CSV_TAX


def taxonomy(name: str) -> dict:
    """Merged taxonomy for a species: seeded table first, then the full CSV."""
    base = SPECIES.get(name, {})
    csvt = _csv_taxonomy().get(name, {})
    return {
        "common": base.get("common", ""),
        "genus": base.get("genus") or csvt.get("genus", ""),
        "family": base.get("family") or csvt.get("family", ""),
        "order": base.get("order") or csvt.get("order", ""),
        "iucn": base.get("iucn") or csvt.get("iucn", "LC"),
    }


def species_key(name: str) -> str:
    """GBIF species key from selected_species.csv (empty if unknown)."""
    return _csv_taxonomy().get(name, {}).get("key", "")


def all_species() -> list[str]:
    """Every species name in selected_species.csv."""
    return list(_csv_taxonomy().keys())


def zone_from_zip(zip_code: str) -> str:
    """Placeholder for a real zip->USDA-zone lookup (phzmapi.org / PRISM raster)."""
    try:
        n = int(zip_code)
    except (TypeError, ValueError):
        n = 0
    zones = ["3b", "4a", "4b", "5a", "5b", "6a", "6b", "7a", "7b", "8a"]
    return zones[n % len(zones)]


def arrange(aspect: str = "sun", zone: str = "6a", area: float = 2500.0,
            toolbox: list[str] | None = None) -> dict:
    """Fill a bed: filter by aspect, seed from the toolbox, greedily cover bloom.

    A deliberately simple version of the constraint-and-scoring 'bed completer'
    the arrange-engine model will replace.
    """
    toolbox = toolbox or []
    if aspect == "sun":
        pool = [g for g in GARDEN if g["sun"] == "sun"]
    elif aspect == "part":
        pool = [g for g in GARDEN if g["sun"] in ("sun", "part")]
    else:
        pool = [g for g in GARDEN if g["sun"] == "shade"]

    chosen: list[dict] = []
    covered: set[int] = set()

    def add(g: dict, why: str) -> None:
        if any(c["sp"] == g["sp"] for c in chosen):
            return
        for m in range(g["bloom"][0], g["bloom"][1] + 1):
            if m:
                covered.add(m)
        chosen.append({**g, "why": why})

    for g in (x for x in pool if x["sp"] in toolbox):
        add(g, "from your toolbox")
    if not any(c["role"] == "structural" for c in chosen):
        for g in pool:
            if g["role"] == "structural":
                add(g, g["note"])
                break
    if not any(c["role"] == "groundcover" for c in chosen):
        for g in pool:
            if g["role"] == "groundcover":
                add(g, "groundcover matrix")
                break

    guard = 0
    while guard < 12 and len(chosen) < 6:
        guard += 1
        best, best_gain = None, 0
        for g in pool:
            if any(c["sp"] == g["sp"] for c in chosen) or g["bloom"][1] == 0:
                continue
            gain = sum(1 for m in range(g["bloom"][0], g["bloom"][1] + 1) if 4 <= m <= 10 and m not in covered)
            if gain > best_gain:
                best, best_gain = g, gain
        if not best or best_gain == 0:
            break
        add(best, f"covers the {MON[best['bloom'][0]-1]}–{MON[best['bloom'][1]-1]} bloom gap")

    drift = max(3, min(7, round(area / 900) * 2 + 1))
    drift = drift if drift % 2 else drift + 1
    for c in chosen:
        c["drift"] = drift

    bloom_months = sorted(m for m in covered if 1 <= m <= 12)
    chosen.sort(key=lambda c: -c["height"])
    return {"aspect": aspect, "zone": zone, "plants": chosen, "bloom_months": bloom_months}
