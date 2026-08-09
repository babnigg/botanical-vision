"""Parse harvested permapeople plan JSON into scoreable layouts.

Each plan gives plant placements (center x,y + canopy width, cm) and embedded
species records. Species traits (layer/height/bloom/color) are normalized into a
per-plan synthetic palette so the v3 metric suite can score real plans — the
calibration check that our metrics aren't circular.

Usage:
    python scripts/permapeople_plans.py --score            # calibration table
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW = PROJECT_DIR / "data" / "permapeople"

_KEYS = {"Layer", "Height", "Width", "Spacing", "Growth", "Light requirement",
         "Water requirement", "Life cycle", "Leaves", "Flowering", "Fruiting",
         "Color", "USDA Hardiness zone", "Soil type", "Root depth", "Edible",
         "Utility", "Native to", "Family", "Warning", "Alternate name",
         "Propagation method", "Edible parts", "Layer(s)"}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_LAYER_MAP = {"Trees": "structural", "Tall trees": "structural", "Shrubs": "structural",
              "Vines": "structural", "Ground cover": "groundcover",
              "Herbs": "seasonal", "Roots": "seasonal"}


def _tags(rec):
    # flat alternating key/value list that degenerates once values go empty
    out = {}
    t = [x for x in (rec.get("tags") or []) if isinstance(x, str)]
    k = 0
    while k < len(t):
        if t[k] in _KEYS and k + 1 < len(t) and t[k + 1] not in _KEYS:
            out[t[k]] = t[k + 1]
            k += 2
        else:
            k += 1
    return out


def _height_cm(txt, layer):
    # PFAF "up to" heights run wild (Echinacea "up to 5m") — clamp per layer
    fallback, lo, hi = {"structural": (200, 100, 600), "seasonal": (60, 20, 150),
                        "groundcover": (10, 5, 30)}[layer]
    m = re.search(r"([\d.]+)\s*(m|cm|ft)", txt or "")
    if not m:
        return fallback
    v = float(m.group(1)) * {"m": 100, "cm": 1, "ft": 30.48}[m.group(2)]
    return min(max(v, lo), hi)


def _bloom(txt):
    if not txt:
        return (0, 0)
    ms = [_MONTHS[w] for w in re.findall(r"[a-z]+", txt.lower()) if w in _MONTHS]
    return (min(ms), max(ms)) if ms else (0, 0)


def _color(txt):
    if not txt:
        return "gray"
    c = txt.split(",")[0].strip().lower()
    try:
        from matplotlib.colors import to_rgb
        to_rgb(c)
        return c
    except ValueError:
        return "gray"


def load_plan(path: Path):
    """-> (pal, plan, w, d, meta) in bvtrain.garden conventions, or None."""
    data = json.loads(path.read_text(encoding="utf-8"))
    species = data.get("species") or {}
    pts = []
    for e in data.get("elements", []):
        ed = e.get("data") or {}
        if e.get("type") != "ellipse" or not e.get("speciesIds") or ed.get("removedDate"):
            continue
        sid = str(e["speciesIds"][0])
        if sid not in species:
            continue
        r = (ed.get("width") or 40) / 200          # cm diameter -> m radius
        pts.append((sid, ed["x"] / 100, ed["y"] / 100, r, ed.get("layer")))
    if len(pts) < 5:
        return None
    pal, idx = [], {}
    edible = 0
    for sid in {p[0] for p in pts}:
        rec = species[sid]
        tg = _tags(rec)
        el_layer = next((p[4] for p in pts if p[0] == sid and p[4]), None)
        layer = _LAYER_MAP.get(el_layer or tg.get("Layer", ""), "seasonal")
        pal.append({"name": rec.get("scientific_name") or rec.get("name", sid),
                    "layer": layer, "h": _height_cm(tg.get("Height"), layer),
                    "s": 40, "sun": 2, "color": _color(tg.get("Color")),
                    "bloom": _bloom(tg.get("Flowering")), "cg": "green",
                    "form": "mound", "tex": "medium", "persist": layer != "seasonal"})
        idx[sid] = len(pal) - 1
        edible += "annual" in (tg.get("Life cycle") or "").lower()   # veg beds run annual
    x0 = min(x - r for _, x, _, r, _ in pts)
    y0 = min(y - r for _, _, y, r, _ in pts)
    plan = [(idx[s], x - x0, y - y0, r) for s, x, y, r, _ in pts]
    w = max(x + r for _, x, _, r in plan)
    d = max(y + r for _, _, y, r in plan)
    meta = {"file": path.name, "n": len(plan), "n_sp": len(pal),
            "edible_frac": edible / len(pal), "w": round(w, 1), "d": round(d, 1)}
    return pal, plan, w, d, meta


def score_real(pal, plan, w, d):
    """v3 metric subset valid without form/texture/sun traits."""
    import sys
    sys.path.insert(0, str(PROJECT_DIR))
    from bvtrain import garden as g
    saved = g.PALETTE
    g.PALETTE = pal
    try:
        # geometry-only subset: embedded records carry no bloom/color/form data
        m = {"overlap": g.m_overlap3(plan), "ground": g.m_ground(plan, w, d),
             "layers": g.m_layers(plan), "drift": g.m_drift3(plan),
             "rhythm": g.m_rhythm(plan, w), "cluster": g.m_cluster(plan)}
    finally:
        g.PALETTE = saved
    m["score"] = sum(m.values()) / len(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--min-plants", type=int, default=15)
    args = ap.parse_args()
    rows = []
    for f in sorted(RAW.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            out = load_plan(f)
        except (KeyError, ValueError, TypeError) as e:
            print(f"skip {f.name}: {e}")
            continue
        if out is None or out[4]["n"] < args.min_plants:
            continue
        pal, plan, w, d, meta = out
        if args.score:
            meta.update(score_real(pal, plan, w, d))
        rows.append(meta)
    if not rows:
        print("no plans parsed")
        return
    if args.score:
        keys = ["overlap", "ground", "layers", "drift", "rhythm", "cluster", "score"]
        orn = [r for r in rows if r["edible_frac"] < 0.4]
        veg = [r for r in rows if r["edible_frac"] >= 0.4]
        print(f"{len(rows)} plans (>= {args.min_plants} plants): "
              f"{len(orn)} ornamental-leaning, {len(veg)} edible-leaning")
        for name, grp in (("ornamental", orn), ("edible", veg)):
            if not grp:
                continue
            print(f"\n{name} means: " + "  ".join(
                f"{k}={sum(r[k] for r in grp) / len(grp):.2f}" for k in keys))
        rows.sort(key=lambda r: -r.get("score", 0))
        for r in rows[:15]:
            print(f"  {r['score']:.2f} n={r['n']:3d} sp={r['n_sp']:2d} "
                  f"{r['w']}x{r['d']}m ed={r['edible_frac']:.1f} {r['file'][:60]}")
    else:
        print(f"{len(rows)} plans parsed")


if __name__ == "__main__":
    main()
