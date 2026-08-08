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

# h/s = mature height/spread (cm); sun: 0 shade, 1 part, 2 full;
# bloom = (first, last) month, (0, 0) = foliage; cg = color group
_P = [
    ("Hydrangea macrophylla", "structural",  150, 150, 1, "cornflowerblue",  (6, 9), "blue"),
    ("Rosa rugosa",           "structural",  150, 120, 2, "palevioletred",   (6, 9), "pink"),
    ("Buxus sempervirens",    "structural",  120, 100, 1, "darkolivegreen",  (0, 0), "green"),
    ("Monarda didyma",        "seasonal",    110,  45, 1, "firebrick",       (7, 9), "red"),
    ("Echinacea purpurea",    "seasonal",    100,  45, 2, "mediumorchid",    (7, 9), "purple"),
    ("Hemerocallis fulva",    "seasonal",     90,  60, 1, "darkorange",      (6, 8), "orange"),
    ("Rudbeckia hirta",       "seasonal",     90,  40, 2, "gold",            (6, 9), "yellow"),
    ("Achillea millefolium",  "seasonal",     70,  45, 2, "seashell",        (6, 9), "white"),
    ("Leucanthemum vulgare",  "seasonal",     60,  40, 2, "white",           (5, 8), "white"),
    ("Salvia nemorosa",       "seasonal",     50,  35, 2, "rebeccapurple",   (5, 9), "purple"),
    ("Alchemilla mollis",     "filler",       40,  50, 1, "yellowgreen",     (5, 7), "yellow"),
    ("Nepeta racemosa",       "filler",       40,  45, 2, "thistle",         (4, 9), "purple"),
    ("Geranium sanguineum",   "filler",       30,  40, 1, "mediumvioletred", (5, 8), "pink"),
    ("Vinca minor",           "groundcover",  10,  60, 0, "slateblue",       (3, 5), "blue"),
    ("Phlox subulata",        "groundcover",  10,  50, 2, "hotpink",         (3, 5), "pink"),
    ("Ajuga reptans",         "groundcover",  10,  40, 0, "midnightblue",    (4, 6), "blue"),
]
PALETTE = [dict(zip(("name", "layer", "h", "s", "sun", "color", "bloom", "cg"), p)) for p in _P]
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


# ── v2: designed corpus (nb 13) ─────────────────────────────────────────────
# elongated drifts, a repeated theme plant, a 60/30/10 color budget, and
# bloom-succession preference — the same rules double as the v2 metrics

def _drift2(plan, i, n, w, d, theta=0.0):
    p = PALETTE[i]
    r = p["s"] / 200
    yc = d * (0.18 + 0.72 * p["h"] / HMAX)
    xc = random.uniform(r * 2, w - r * 2)
    ca, sa = math.cos(theta), math.sin(theta)
    placed = 0
    for _ in range(n * 30):
        if placed == n:
            break
        u = random.gauss(0, r * 2.8)
        v = random.gauss(0, r * 0.9)
        x, y = xc + u * ca - v * sa, yc + u * sa + v * ca
        if _fits(plan, x, y, r, w, d):
            plan.append((i, x, y, r))
            placed += 1
    if 0 < placed < n:
        del plan[-placed:]


def gen_plan2(w, d, sun=None):
    plan, used = [], set()
    pool_all = [i for i in range(len(PALETTE))
                if sun is None or abs(PALETTE[i]["sun"] - sun) <= 1]
    groups = list({PALETTE[i]["cg"] for i in pool_all if PALETTE[i]["cg"] != "green"})
    random.shuffle(groups)
    dom, sec, acc = (groups * 3)[:3]
    target = {dom: 0.6, sec: 0.3, acc: 0.1}
    garea = {}
    bloomed = set()

    def bloom_gain(i):
        b = PALETTE[i]["bloom"]
        return len(set(range(b[0], b[1] + 1)) - bloomed) if b[1] else 0

    def pick(layer):
        pool = [i for i in pool_all if PALETTE[i]["layer"] == layer] or BY_LAYER[layer]
        fresh = [i for i in pool if i not in used] or pool
        # steer toward the color group furthest below its 60/30/10 area share
        tot = sum(garea.values()) or 1.0
        cg = max(target, key=lambda c: target[c] - garea.get(c, 0.0) / tot)
        cands = [i for i in fresh if PALETTE[i]["cg"] == cg] or fresh
        best = max(bloom_gain(i) for i in cands)
        cands = [i for i in cands if bloom_gain(i) == best] or cands
        i = random.choice(cands)
        used.add(i)
        garea[PALETTE[i]["cg"]] = garea.get(PALETTE[i]["cg"], 0.0) + (PALETTE[i]["s"] / 200) ** 2
        b = PALETTE[i]["bloom"]
        if b[1]:
            bloomed.update(range(b[0], b[1] + 1))
        return i

    for _ in range(max(1, round(w / 3))):
        _drift2(plan, pick("structural"), 1, w, d)
    theme = pick("seasonal")
    for _ in range(max(2, round(w / 2.5))):          # theme plant repeats across the bed
        _drift2(plan, theme, random.choice((3, 5)), w, d, random.gauss(0, 0.3))
    for _ in range(round(w * 0.7)):
        _drift2(plan, pick(random.choices(("seasonal", "filler"), (0.7, 0.3))[0]),
                random.choice((3, 5, 7)), w, d, random.gauss(0, 0.3))
    for _ in range(min(3, round(w / 2))):
        _drift2(plan, pick("groundcover"), random.choice((3, 5)), w, d, random.gauss(0, 0.3))
    return plan


def m_color(plan):
    # canopy area per color group vs a 60/30/10 split of the top three
    if not plan:
        return 0.0
    area = {}
    for i, _, _, r in plan:
        if PALETTE[i]["cg"] != "green":      # foliage is a neutral backdrop, not a color block
            area[PALETTE[i]["cg"]] = area.get(PALETTE[i]["cg"], 0) + math.pi * r * r
    if not area:
        return 0.0
    tot = sum(area.values())
    top = sorted(area.values(), reverse=True)[:3] + [0.0, 0.0, 0.0]
    return max(0.0, 1 - sum(abs(a / tot - t) for a, t in zip(top[:3], (0.6, 0.3, 0.1))))


def m_bloom(plan):
    # fraction of apr-oct with at least one species in bloom
    months = set()
    for i, _, _, _ in plan:
        b = PALETTE[i]["bloom"]
        if b[1]:
            months.update(range(b[0], b[1] + 1))
    return len(months & set(range(4, 11))) / 7


def score2(plan, w, d, sun=None):
    m = score(plan, w, d, sun)
    m.pop("score")
    m["color"] = m_color(plan)
    m["bloom"] = m_bloom(plan)
    m["score"] = float(np.mean(list(m.values())))
    return m


def repair(plan, w, d):
    # push overlapping plants apart, clamp to bed, drop one plant from even drifts
    plan = [list(p) for p in plan]
    for _ in range(40):
        moved = False
        for a in range(len(plan)):
            for b in range(a + 1, len(plan)):
                _, xa, ya, ra = plan[a]
                _, xb, yb, rb = plan[b]
                dx, dy = xb - xa, yb - ya
                dist = math.hypot(dx, dy) or 1e-6
                need = 0.78 * (ra + rb)
                if dist < need:
                    push = (need - dist) / 2 + 1e-3
                    plan[a][1] -= dx / dist * push
                    plan[a][2] -= dy / dist * push
                    plan[b][1] += dx / dist * push
                    plan[b][2] += dy / dist * push
                    moved = True
        for p in plan:
            p[1] = min(max(p[1], p[3] * 0.6), w - p[3] * 0.6)
            p[2] = min(max(p[2], p[3] * 0.4), d - p[3] * 0.4)
        if not moved:
            break
    counts = Counter(p[0] for p in plan if PALETTE[p[0]]["layer"] != "structural")
    for sp, n in counts.items():
        if n >= 2 and n % 2 == 0:
            k = max(i for i, p in enumerate(plan) if p[0] == sp)
            del plan[k]
    return [tuple(p) for p in plan]


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
