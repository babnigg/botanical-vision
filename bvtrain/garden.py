"""shared primitives for the compose notebooks (10-12): plant palette, rule
generator, metrics, plan drawing. models and tokenizers stay in the notebooks."""
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# h/s = mature height/spread (cm); sun: 0 shade, 1 part, 2 full
_P = [
    ("Hydrangea macrophylla", "structural",  150, 150, 1, "cornflowerblue"),
    ("Rosa rugosa",           "structural",  150, 120, 2, "palevioletred"),
    ("Buxus sempervirens",    "structural",  120, 100, 1, "darkolivegreen"),
    ("Monarda didyma",        "seasonal",    110,  45, 1, "firebrick"),
    ("Echinacea purpurea",    "seasonal",    100,  45, 2, "mediumorchid"),
    ("Hemerocallis fulva",    "seasonal",     90,  60, 1, "darkorange"),
    ("Rudbeckia hirta",       "seasonal",     90,  40, 2, "gold"),
    ("Achillea millefolium",  "seasonal",     70,  45, 2, "seashell"),
    ("Leucanthemum vulgare",  "seasonal",     60,  40, 2, "white"),
    ("Salvia nemorosa",       "seasonal",     50,  35, 2, "rebeccapurple"),
    ("Alchemilla mollis",     "filler",       40,  50, 1, "yellowgreen"),
    ("Nepeta racemosa",       "filler",       40,  45, 2, "thistle"),
    ("Geranium sanguineum",   "filler",       30,  40, 1, "mediumvioletred"),
    ("Vinca minor",           "groundcover",  10,  60, 0, "slateblue"),
    ("Phlox subulata",        "groundcover",  10,  50, 2, "hotpink"),
    ("Ajuga reptans",         "groundcover",  10,  40, 0, "midnightblue"),
]
PALETTE = [dict(zip(("name", "layer", "h", "s", "sun", "color"), p)) for p in _P]
BY_LAYER = {lay: [i for i, p in enumerate(PALETTE) if p["layer"] == lay]
            for lay in ("structural", "seasonal", "filler", "groundcover")}
HMAX = max(p["h"] for p in PALETTE)
SUN_NAMES = ("shade", "part sun", "full sun")


# ── rule generator ──────────────────────────────────────────────────────────
# a plan is [(species idx, x, y, r)] in meters; y=0 is the front edge

def _fits(plan, x, y, r, w, d):
    if not (r * 0.6 < x < w - r * 0.6 and r * 0.4 < y < d - r * 0.4):
        return False
    return all((x - px) ** 2 + (y - py) ** 2 >= (0.75 * (r + pr)) ** 2
               for _, px, py, pr in plan)


def _drift(plan, i, n, w, d):
    # n of one species clustered on a random anchor, depth set by height
    p = PALETTE[i]
    r = p["s"] / 200
    yc = d * (0.18 + 0.72 * p["h"] / HMAX)
    xc = random.uniform(r, w - r)
    placed = 0
    for _ in range(n * 25):
        if placed == n:
            break
        x = random.gauss(xc, r * 2.2)
        y = random.gauss(yc, max(r, d * 0.06))
        if _fits(plan, x, y, r, w, d):
            plan.append((i, x, y, r))
            placed += 1
    if 0 < placed < n:          # keep drifts odd — drop partials
        del plan[-placed:]


def gen_plan(w, d, sun=None):
    # sun None = ignore; else species must be within one sun level of the bed
    plan, used = [], set()

    def pick(layer):
        pool = [i for i in BY_LAYER[layer]
                if sun is None or abs(PALETTE[i]["sun"] - sun) <= 1]
        fresh = [i for i in pool if i not in used]
        i = random.choice(fresh or pool)
        used.add(i)
        return i

    for _ in range(max(1, round(w / 3))):
        _drift(plan, pick("structural"), 1, w, d)
    for _ in range(round(w * 0.9)):
        _drift(plan, pick(random.choices(("seasonal", "filler"), (0.7, 0.3))[0]),
               random.choice((3, 5, 7)), w, d)
    for _ in range(min(3, round(w / 2))):
        _drift(plan, pick("groundcover"), random.choice((3, 5)), w, d)
    return plan


def gen_random(w, d, sun=None):
    plan = []
    for _ in range(random.randint(20, 40)):
        i = random.randrange(len(PALETTE))
        plan.append((i, random.uniform(0, w), random.uniform(0, d), PALETTE[i]["s"] / 200))
    return plan


def rand_bed():
    return round(random.uniform(3.5, 7.5), 1), round(random.uniform(1.8, 3.0), 1)


def rand_site():
    w, d = rand_bed()
    return w, d, random.randrange(3)


# ── metrics (1 = clean) ─────────────────────────────────────────────────────

def m_overlap(plan):
    if len(plan) < 2:
        return 1.0
    bad = tot = 0
    for a in range(len(plan)):
        for b in range(a + 1, len(plan)):
            _, xa, ya, ra = plan[a]
            _, xb, yb, rb = plan[b]
            tot += 1
            if math.hypot(xa - xb, ya - yb) < 0.75 * (ra + rb):
                bad += 1
    return 1 - bad / tot


def m_height(plan):
    # taller plants shouldn't stand in front of shorter ones in the same corridor
    bad = tot = 0
    for ia, xa, ya, _ in plan:
        for ib, xb, yb, _ in plan:
            if abs(xa - xb) < 1.0 and ya < yb - 0.4:
                tot += 1
                if PALETTE[ia]["h"] > PALETTE[ib]["h"] + 30:
                    bad += 1
    return 1 - bad / tot if tot else 1.0


def m_cover(plan, w, d):
    a = sum(math.pi * r * r for _, _, _, r in plan) / (w * d)
    return max(0.0, 1 - abs(a - 0.75) / 0.75)


def m_drift(plan):
    c = Counter(i for i, _, _, _ in plan if PALETTE[i]["layer"] != "structural")
    if not c:
        return 0.0
    return sum(1 for n in c.values() if n >= 3 and n % 2 == 1) / len(c)


def m_sun(plan, sun):
    if not plan:
        return 0.0
    return sum(1 for i, _, _, _ in plan if abs(PALETTE[i]["sun"] - sun) <= 1) / len(plan)


def score(plan, w, d, sun=None):
    m = {"overlap": m_overlap(plan), "height": m_height(plan),
         "cover": m_cover(plan, w, d), "drift": m_drift(plan)}
    if sun is not None:
        m["sun"] = m_sun(plan, sun)
    m["score"] = float(np.mean(list(m.values())))
    return m


# ── drawing ─────────────────────────────────────────────────────────────────

def show_plan(plan, w, d, ax, title="", pins=()):
    ax.set_facecolor("linen")
    ax.add_patch(Rectangle((0, 0), w, d, facecolor="wheat", edgecolor="sienna"))
    for k, (i, x, y, r) in enumerate(plan):
        pinned = k in pins
        ax.add_patch(Circle((x, y), r, facecolor=PALETTE[i]["color"],
                            edgecolor="black" if pinned else "sienna",
                            lw=1.8 if pinned else 0.6, alpha=0.85))
    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, d + 0.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def seg_image(plan, w, d, px=100, pad=0.5):
    # ade20k-ish colors so a pretrained seg controlnet reads it semantically
    img = Image.new("RGB", (int((w + 2 * pad) * px), int((d + 2 * pad) * px)), (4, 250, 7))
    dr = ImageDraw.Draw(img)
    dr.rectangle([pad * px, pad * px, (pad + w) * px, (pad + d) * px], fill=(120, 120, 70))
    lay_col = {"structural": (4, 200, 3), "seasonal": (255, 5, 153),
               "filler": (255, 163, 0), "groundcover": (10, 255, 71)}
    for i, x, y, r in plan:
        cx, cy = (pad + x) * px, (pad + d - y) * px      # image y runs front-down
        dr.ellipse([cx - r * px, cy - r * px, cx + r * px, cy + r * px],
                   fill=lay_col[PALETTE[i]["layer"]])
    return img


def ckpt_dir():
    # mirror train.checkpoint_dir: drive on colab, /kaggle/working on kaggle, ../checkpoints locally
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        p = Path("/content/drive/MyDrive/botanical-vision/checkpoints")
    except ImportError:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            p = Path("/kaggle/working")
        else:
            p = Path(__file__).resolve().parent.parent / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p
