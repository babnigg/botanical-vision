"""
Download flowering-plant images from iNaturalist, driven by data/selected_species.csv.

The GBIF species list gives us WHICH species to collect (by scientific name).
iNaturalist gives us the actual photos. GBIF and iNat use different taxonomy keys,
so for each species we first resolve its scientific name -> iNat taxon_id, then
download research-grade photos for that taxon.

Rate limiting: calls to api.inaturalist.org are kept sequential and delayed
(~85 req/min, under the 100/min limit). Image files come from the iNat CDN
(static.inaturalist.org), which we fetch concurrently with a thread pool.

Usage:
    python scripts/download_inaturalist.py --images_per_species 100 --workers 8

Resumable: skips species/images already on disk. Safe to Ctrl-C and rerun.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_BASE = "https://api.inaturalist.org/v1"
PROJECT_DIR = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_DIR / "data" / "selected_species.csv"
IMAGE_DIR = PROJECT_DIR / "data" / "raw" / "images"
RESOLVED_PATH = PROJECT_DIR / "data" / "inat_taxon_map.json"
API_DELAY = 0.7  # seconds between API calls (~85 req/min)

session = requests.Session()
session.headers.update({"User-Agent": "botanical-vision-eda/1.0 (grad CV coursework)"})


def resolve_taxon_id(scientific_name: str) -> int | None:
    """Look up the iNaturalist taxon_id for a scientific name (plants only).

    `q` is a fuzzy search, so we verify the returned taxon's name actually matches the
    query (case-insensitive) before trusting it - otherwise a synonym/homonym could
    silently resolve to a different species and mislabel every downloaded photo.
    """
    params = {"q": scientific_name, "rank": "species", "iconic_taxa": "Plantae", "per_page": 5}
    try:
        resp = session.get(f"{API_BASE}/taxa", params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except requests.RequestException:
        return None
    want = scientific_name.strip().lower()
    for r in results:
        if r.get("name", "").strip().lower() == want:
            return r.get("id")
    return None  # no exact name match -> leave unresolved rather than mislabel


def collect_photo_urls(taxon_id: int, num_needed: int, seen: set[str]) -> list[tuple[str, str]]:
    """Page through observations (sequential API calls) to gather (photo_id, url) pairs."""
    pairs: list[tuple[str, str]] = []
    page = 1
    while len(pairs) < num_needed and page <= 10:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 200,
            "page": page,
            "order": "desc",
            "order_by": "votes",
        }
        try:
            resp = session.get(f"{API_BASE}/observations", params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except requests.RequestException:
            break
        if not results:
            break
        for obs in results:
            for photo in obs.get("photos", []):
                pid = str(photo.get("id", ""))
                if not pid or pid in seen:
                    continue
                url = photo.get("url", "").replace("/square.", "/medium.")
                if url:
                    seen.add(pid)
                    pairs.append((pid, url))
        page += 1
        time.sleep(API_DELAY)
    return pairs[:num_needed]


def fetch_one(args: tuple[str, str, Path]) -> bool:
    """Download a single image file atomically. Runs in a worker thread.

    Writes to a .tmp file then os.replace()s into place, so a process killed
    mid-download never leaves a truncated .jpg that a later resume would treat
    as complete.
    """
    pid, url, out_dir = args
    species_key = out_dir.name.split("_")[0]
    final = out_dir / f"{species_key}_{pid}.jpg"
    tmp = out_dir / f"{species_key}_{pid}.jpg.tmp"
    try:
        img = session.get(url, timeout=20)
        img.raise_for_status()
        content = img.content
        # Basic sanity: non-trivial size and JPEG magic bytes
        if len(content) < 1024 or content[:2] != b"\xff\xd8":
            return False
        tmp.write_bytes(content)
        os.replace(tmp, final)  # atomic on same filesystem
        return True
    except requests.RequestException:
        tmp.unlink(missing_ok=True)
        return False
    except OSError:
        tmp.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download iNaturalist images for selected species")
    parser.add_argument("--images_per_species", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="concurrent image downloads")
    parser.add_argument("--limit", type=int, default=None, help="only process first N species (testing)")
    args = parser.parse_args()

    if not SELECTED_CSV.exists():
        raise SystemExit(f"Missing {SELECTED_CSV}. Run notebooks/01_eda_species.ipynb first.")

    species = pd.read_csv(SELECTED_CSV)
    if args.limit is not None:
        species = species.head(args.limit)
    print(f"Processing {len(species)} species, up to {args.images_per_species} images each, {args.workers} workers.")

    # Clean up any stray .tmp files from a previously interrupted run
    stray = list(IMAGE_DIR.glob("**/*.jpg.tmp"))
    for f in stray:
        f.unlink(missing_ok=True)
    if stray:
        print(f"Cleaned {len(stray)} partial temp files from a prior run.")

    taxon_map = json.loads(RESOLVED_PATH.read_text()) if RESOLVED_PATH.exists() else {}
    total, done_species, unresolved = 0, 0, []

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (_, row) in enumerate(tqdm(species.iterrows(), total=len(species), desc="Species")):
                name = row["species"]
                safe = str(name).replace(" ", "_").replace("/", "_")
                species_key = int(row["speciesKey"])  # guard against float formatting
                out_dir = IMAGE_DIR / f"{species_key}_{safe}"
                out_dir.mkdir(parents=True, exist_ok=True)

                existing = {f.stem.split("_")[-1] for f in out_dir.glob("*.jpg")}
                if len(existing) >= args.images_per_species:
                    total += len(existing)
                    done_species += 1
                    continue

                # Resolve name -> iNat taxon_id (cached across runs, incl. None misses)
                if name in taxon_map:
                    taxon_id = taxon_map[name]
                else:
                    taxon_id = resolve_taxon_id(name)
                    taxon_map[name] = taxon_id
                    RESOLVED_PATH.write_text(json.dumps(taxon_map, indent=2))
                    time.sleep(API_DELAY)
                if not taxon_id:
                    unresolved.append(name)
                    continue

                need = args.images_per_species - len(existing)
                pairs = collect_photo_urls(taxon_id, need, set(existing))
                jobs = [(pid, url, out_dir) for pid, url in pairs]
                got = sum(f.result() for f in as_completed(pool.submit(fetch_one, j) for j in jobs))
                total += len(existing) + got
                done_species += 1

                # Periodic heartbeat for overnight runs
                if (i + 1) % 100 == 0:
                    tqdm.write(f"  [{i+1}/{len(species)}] {total:,} images so far, {len(unresolved)} unresolved")
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Rerun the same command to resume.")
    finally:
        RESOLVED_PATH.write_text(json.dumps(taxon_map, indent=2))

    print(f"\nDone. {total:,} images across {done_species} species ({len(unresolved)} unresolved).")
    if unresolved:
        print(f"Unresolved on iNat ({len(unresolved)}): {unresolved[:10]}{'...' if len(unresolved) > 10 else ''}")
    print(f"Images: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
