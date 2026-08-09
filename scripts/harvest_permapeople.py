"""Harvest structured garden plans from permapeople.org.

Plan pages expose an unauthenticated JSON endpoint /{user}/plans/{slug}/data:
plant placements as centered ellipses (cm), bed rectangles/polygons, and the
full species records embedded. Saves raw JSON + the page's og:image snapshot to
data/permapeople/ (gitignored). Resumable: existing files are skipped.

Data is CC BY-SA (plant db) / community plans — local research use, not
redistributed. ~1 req/s to be polite.

Usage:
    python scripts/harvest_permapeople.py [--pages 14]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "data" / "permapeople"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) academic research",
        "Accept": "application/json"}


def list_plans(pages: int) -> list[tuple[str, str]]:
    seen, out = set(), []
    for n in range(1, pages + 1):
        r = requests.get(f"https://permapeople.org/plans?page={n}", headers={**HDRS, "Accept": "text/html"}, timeout=30)
        r.raise_for_status()
        for user, slug in re.findall(r'href="/([^"/]+)/plans/([^"/]+)"', r.text):
            if (user, slug) not in seen:
                seen.add((user, slug))
                out.append((user, slug))
        time.sleep(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=14)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    plans = list_plans(args.pages)
    print(f"{len(plans)} plans listed")
    manifest = []
    for k, (user, slug) in enumerate(plans):
        stem = f"{user}__{slug}"[:120].replace("%", "_")
        jf = OUT / f"{stem}.json"
        row = {"user": user, "slug": slug, "file": jf.name, "ok": False}
        if jf.exists():
            row["ok"] = True
            manifest.append(row)
            continue
        try:
            r = requests.get(f"https://permapeople.org/{user}/plans/{slug}/data", headers=HDRS, timeout=30)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
                jf.write_text(json.dumps(data), encoding="utf-8")
                row["ok"] = True
                row["n_plants"] = sum(1 for e in data.get("elements", [])
                                      if e.get("type") == "ellipse" and e.get("speciesIds"))
            time.sleep(1)
            # snapshot render (og:image) for render/layout pairs
            pg = requests.get(f"https://permapeople.org/{user}/plans/{slug}", headers={**HDRS, "Accept": "text/html"}, timeout=30)
            m = re.search(r'property="og:image" content="([^"]+)"', pg.text)
            if m and "cdn" in m.group(1):
                img = requests.get(m.group(1), headers=HDRS, timeout=30)
                if img.status_code == 200 and len(img.content) > 5000:
                    (OUT / f"{stem}.jpg").write_bytes(img.content)
        except requests.RequestException as e:
            row["err"] = str(e)[:80]
        manifest.append(row)
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(plans)} fetched")
        time.sleep(1)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ok = sum(1 for r in manifest if r["ok"])
    print(f"done: {ok}/{len(manifest)} plans saved -> {OUT}")


if __name__ == "__main__":
    main()
