"""Harvest 2D planting-plan drawings into data/real_plans/ (gitignored, local-only).

Sources: gardengatemagazine.com garden-plans articles and almanac.com's layout
library — publisher pages whose plan illustrations match the circle-symbol genre
notebook 14 extracts from. Every download is recorded in data/real_plans/sources.csv
(filename, image url, page url, source); the collection stays out of git because the
images are publisher-owned. Notebook 14's ensemble gate curates whatever lands here,
so over-collecting is fine. Rerun anytime; downloads dedupe by content hash.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "data" / "real_plans"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "sources.csv"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def save(session, url, page, source, prefix, rows, seen):
    try:
        r = session.get(url, timeout=25)
        if r.status_code != 200 or len(r.content) < 20_000:
            return 0
        h = hashlib.md5(r.content).hexdigest()
        if h in seen:
            return 0
        img = Image.open(io.BytesIO(r.content))
        img.load()
        if img.width < 500 or img.height < 350:
            return 0
        seen.add(h)
        ext = (img.format or "JPEG").lower().replace("jpeg", "jpg")
        name = f"{prefix}_{h[:12]}.{ext}"
        (OUT / name).write_bytes(r.content)
        rows.append([name, url, page, source])
        return 1
    except Exception:
        return 0


def garden_gate(session, rows, seen):
    base = "https://www.gardengatemagazine.com"
    hub = session.get(f"{base}/articles/garden-plans/", timeout=25).text
    cats = sorted(set(re.findall(r'"(/articles/garden-plans/[a-z-]+/)"', hub)))
    articles = set()
    for c in cats + ["/articles/garden-plans/all/"]:
        try:
            r = session.get(base + c, timeout=25)
            articles.update(re.findall(r'"(/articles/garden-plans/[a-z-]+/[a-z0-9-]+/)"', r.text))
        except Exception:
            continue
        time.sleep(1.0)
    n = 0
    seen_asset = set()
    for a in sorted(articles):
        try:
            r = session.get(base + a, timeout=25)
        except Exception:
            continue
        for u in re.findall(r'(?:src|data-src|srcset)="(//images\.ctfassets\.net/[^"\s]+)"', r.text):
            m = re.match(r"//images\.ctfassets\.net/[^/]+/([^/]+)/", u)
            if not m or m.group(1) in seen_asset:
                continue
            seen_asset.add(m.group(1))
            n += save(session, "https:" + u.split("?")[0] + "?w=1400", base + a,
                      "gardengate garden-plans", "gg", rows, seen)
        time.sleep(1.0)
    print(f"garden gate: {n} images from {len(articles)} articles")


def almanac(session, rows, seen):
    base = "https://www.almanac.com"
    hub = session.get(f"{base}/free-garden-layouts-plans-library", timeout=25).text
    pages = sorted(set(re.findall(r'href="(/[^"]*(?:plan|layout)[^"]*)"', hub)))
    pages = [p for p in pages if "calculat" not in p and "calendar" not in p]
    n = 0
    seen_url = set()
    for pg in pages:
        try:
            r = session.get(base + pg, timeout=25)
        except Exception:
            continue
        for u in re.findall(r'(?:src|data-src)="(https://www\.almanac\.com/sites/default/files/'
                            r'[^"]+\.(?:png|jpg|jpeg|gif))', r.text, re.I):
            if u in seen_url:
                continue
            seen_url.add(u)
            n += save(session, u, base + pg, "almanac layouts", "alm", rows, seen)
        time.sleep(1.0)
    print(f"almanac: {n} images from {len(pages)} pages")


def main():
    session = requests.Session()
    session.headers.update(HDRS)
    rows = list(csv.reader(open(MANIFEST, encoding="utf-8"))) if MANIFEST.exists() else []
    seen = set()
    for p in OUT.iterdir():
        if p.suffix.lower() in (".jpg", ".png", ".gif", ".jpeg"):
            seen.add(hashlib.md5(p.read_bytes()).hexdigest())
    garden_gate(session, rows, seen)
    almanac(session, rows, seen)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("manifest rows:", len(rows))


if __name__ == "__main__":
    main()
