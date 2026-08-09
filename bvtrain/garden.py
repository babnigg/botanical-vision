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
# bloom = (first, last) month, (0, 0) = foliage; cg = color group;
# form = Oudolf flowerhead taxonomy; tex = foliage texture (UF/IFAS fine/medium/coarse);
# persist = holds structure into autumn/winter (seedheads / evergreen)
_P = [
    ("Hydrangea macrophylla", "structural",  150, 150, 1, "cornflowerblue",  (6, 9), "blue",   "globe",  "coarse", True),
    ("Rosa rugosa",           "structural",  150, 120, 2, "palevioletred",   (6, 9), "pink",   "globe",  "coarse", True),
    ("Buxus sempervirens",    "structural",  120, 100, 1, "darkolivegreen",  (0, 0), "green",  "mound",  "fine",   True),
    ("Monarda didyma",        "seasonal",    110,  45, 1, "firebrick",       (7, 9), "red",    "globe",  "medium", True),
    ("Echinacea purpurea",    "seasonal",    100,  45, 2, "mediumorchid",    (7, 9), "purple", "daisy",  "medium", True),
    ("Hemerocallis fulva",    "seasonal",     90,  60, 1, "darkorange",      (6, 8), "orange", "mound",  "medium", False),
    ("Rudbeckia hirta",       "seasonal",     90,  40, 2, "gold",            (6, 9), "yellow", "daisy",  "medium", True),
    ("Achillea millefolium",  "seasonal",     70,  45, 2, "seashell",        (6, 9), "white",  "umbel",  "fine",   True),
    ("Leucanthemum vulgare",  "seasonal",     60,  40, 2, "white",           (5, 8), "white",  "daisy",  "medium", False),
    ("Salvia nemorosa",       "seasonal",     50,  35, 2, "rebeccapurple",   (5, 9), "purple", "spire",  "fine",   True),
    ("Alchemilla mollis",     "filler",       40,  50, 1, "yellowgreen",     (5, 7), "yellow", "mound",  "medium", False),
    ("Nepeta racemosa",       "filler",       40,  45, 2, "thistle",         (4, 9), "purple", "mound",  "fine",   False),
    ("Geranium sanguineum",   "filler",       30,  40, 1, "mediumvioletred", (5, 8), "pink",   "mound",  "fine",   False),
    ("Vinca minor",           "groundcover",  10,  60, 0, "slateblue",       (3, 5), "blue",   "carpet", "fine",   True),
    ("Phlox subulata",        "groundcover",  10,  50, 2, "hotpink",         (3, 5), "pink",   "carpet", "fine",   True),
    ("Ajuga reptans",         "groundcover",  10,  40, 0, "midnightblue",    (4, 6), "blue",   "carpet", "medium", True),
    # v3 additions (appended — indices 0-15 keep matching pre-v3 checkpoints):
    # fill the missing forms (spire/plume/screen) and the aug-oct bloom gap
    ("Veronica spicata",      "seasonal",     45,  30, 2, "royalblue",       (6, 8), "blue",   "spire",  "fine",   True),
    ("Astilbe japonica",      "seasonal",     60,  45, 0, "lightcoral",      (6, 7), "pink",   "plume",  "fine",   True),
    ("Calamagrostis acutiflora", "structural", 150, 60, 2, "tan",            (6, 10), "green", "screen", "fine",   True),
    ("Sedum telephium",       "seasonal",     50,  45, 2, "indianred",       (8, 10), "pink",  "umbel",  "coarse", True),
    ("Symphyotrichum novae-angliae", "seasonal", 120, 60, 2, "mediumpurple", (8, 10), "purple", "daisy", "medium", False),
]
PALETTE = [dict(zip(("name", "layer", "h", "s", "sun", "color", "bloom", "cg",
                     "form", "tex", "persist"), p)) for p in _P]
COMMON = {  # colloquial names for the demo UI
    'Hydrangea macrophylla': 'bigleaf hydrangea',
    'Rosa rugosa': 'beach rose',
    'Buxus sempervirens': 'boxwood',
    'Monarda didyma': 'bee balm',
    'Echinacea purpurea': 'purple coneflower',
    'Hemerocallis fulva': 'daylily',
    'Rudbeckia hirta': 'black-eyed susan',
    'Achillea millefolium': 'yarrow',
    'Leucanthemum vulgare': 'oxeye daisy',
    'Salvia nemorosa': 'meadow sage',
    'Alchemilla mollis': "lady's mantle",
    'Nepeta racemosa': 'catmint',
    'Geranium sanguineum': 'cranesbill',
    'Vinca minor': 'periwinkle',
    'Phlox subulata': 'creeping phlox',
    'Ajuga reptans': 'bugleweed',
    'Veronica spicata': 'spike speedwell',
    'Astilbe japonica': 'astilbe',
    'Calamagrostis acutiflora': 'feather reed grass',
    'Sedum telephium': 'autumn stonecrop',
    'Symphyotrichum novae-angliae': 'new england aster',
}
for _pp in PALETTE:
    _pp["common"] = COMMON[_pp["name"]]
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

    def pick(layer, max_s=None):
        pool = [i for i in pool_all if PALETTE[i]["layer"] == layer] or BY_LAYER[layer]
        if max_s:
            pool = [i for i in pool if PALETTE[i]["s"] <= max_s] or pool
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


# ── v3: layered corpus (nb 15) ──────────────────────────────────────────────
# research-grounded rules: layers are strata — groundcover carpets the bed
# *under* taller plants (Rainer & West), so overlap only counts within a
# stratum. drifts are elongated and oblique (Jekyll), theme plants repeat at
# quasi-regular intervals (Oudolf), and the metric suite comes from independent
# literature (layer shares, form/texture contrast, seasonal min, Moon-Spencer
# hue harmony, per-species clustering) rather than the generator's own rules.

STRATUM_GAP = 40   # cm height gap above which two plants occupy different strata


def _conflict(ia, ib):
    pa, pb = PALETTE[ia], PALETTE[ib]
    if ia == ib and pa["layer"] == "groundcover":   # a mat knits — pack freely
        return False
    return pa["layer"] == pb["layer"] or abs(pa["h"] - pb["h"]) < STRATUM_GAP


def _fits3(plan, i, x, y, r, w, d):
    if not (r * 0.6 < x < w - r * 0.6 and r * 0.4 < y < d - r * 0.4):
        return False
    return all(not _conflict(i, j) or (x - px) ** 2 + (y - py) ** 2 >= (0.75 * (r + pr)) ** 2
               for j, px, py, pr in plan)


def _drift3(plan, i, n, w, d, xc=None):
    # n plants strung along an oblique axis with light jitter — elongated by
    # construction; the angle flattens on shallow beds so the drift fits
    p = PALETTE[i]
    r = p["s"] / 200
    step = r * 1.7
    sa = math.sin(math.radians(random.uniform(20, 55)))
    if n > 1:
        sa = min(sa, d * 0.45 / ((n - 1) * step))
    sa *= random.choice((-1, 1))
    ca = math.sqrt(1 - sa * sa)
    xc = random.uniform(r * 2, w - r * 2) if xc is None else xc
    yc = d * (0.15 + 0.72 * p["h"] / HMAX)
    placed = 0
    for k in range(n):
        u = (k - (n - 1) / 2) * step
        for _ in range(12):
            uu = u + random.gauss(0, r * 0.3)
            vv = random.gauss(0, r * 0.45)
            x, y = xc + uu * ca - vv * sa, yc + uu * sa + vv * ca
            if _fits3(plan, i, x, y, r, w, d):
                plan.append((i, x, y, r))
                placed += 1
                break
    if 0 < placed < min(n, 3):
        del plan[-placed:]
        placed = 0
    return placed


def gen_plan3(w, d, sun=None):
    plan = []
    pool_all = [i for i in range(len(PALETTE))
                if sun is None or abs(PALETTE[i]["sun"] - sun) <= 1]
    used = set()
    groups = list({PALETTE[i]["cg"] for i in pool_all if PALETTE[i]["cg"] != "green"})
    random.shuffle(groups)
    dom, sec, acc = (groups * 3)[:3]
    target = {dom: 0.6, sec: 0.3, acc: 0.1}
    garea, bloomed = {}, set()

    def bloom_gain(i):
        b = PALETTE[i]["bloom"]
        return len(set(range(b[0], b[1] + 1)) - bloomed) if b[1] else 0

    def pick(layer, max_s=None):
        pool = [i for i in pool_all if PALETTE[i]["layer"] == layer] or BY_LAYER[layer]
        if max_s:
            pool = [i for i in pool if PALETTE[i]["s"] <= max_s] or pool
        fresh = [i for i in pool if i not in used] or pool
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

    # structural: sparse singles spaced across the back
    ns = max(1, round(w / 3))
    for k in range(ns):
        _drift3(plan, pick("structural"), 1, w, d,
                xc=w * (k + 0.5 + random.uniform(-0.2, 0.2)) / ns)
    # theme plant: repeated at quasi-regular intervals along the bed (rhythm) —
    # compact 3s clamped inside slots so repeats read as separate groups, not a band
    theme = pick("seasonal", max_s=55)
    reps = max(3, round(w / 2.0))
    slot = w / reps
    ext = PALETTE[theme]["s"] / 200 * 3.4      # drift x-extent estimate
    for j in range(reps):
        lo, hi = j * slot + ext / 2 + 0.45, (j + 1) * slot - ext / 2 - 0.45
        for _ in range(3):
            if _drift3(plan, theme, 3, w, d,
                       xc=random.uniform(min(lo, hi), max(lo, hi))) >= 3:
                break
    # supporting seasonal + filler drifts
    for _ in range(round(w * 0.8)):
        _drift3(plan, pick(random.choices(("seasonal", "filler"), (0.75, 0.25))[0]),
                random.choice((3, 5, 7)), w, d)
    # groundcover: carpet the whole bed in 1-2 sweeps (green mulch, ~100% cover)
    ngc = random.choice((1, 2))
    cut = w * random.uniform(0.35, 0.65)
    bands = [(0, cut), (cut, w)] if ngc == 2 else [(0, w)]
    for x0, x1 in bands:
        gi = pick("groundcover")
        r = PALETTE[gi]["s"] / 200
        step = r * 1.65
        nx, ny = max(1, math.ceil((x1 - x0) / step)), max(1, math.ceil(d / step))
        for b in range(ny):
            for a in range(nx):
                # hex-offset rows: square grids of circles cap out at pi/4 cover
                x = x0 + (a + 0.5 + 0.5 * (b % 2)) * (x1 - x0) / nx + random.gauss(0, r * 0.2)
                y = (b + 0.5) * d / ny + random.gauss(0, r * 0.2)
                if _fits3(plan, gi, x, y, r, w, d):
                    plan.append((gi, x, y, r))
    return plan


def _clusters(plan, link=0.2):
    # same-species points chained when gaps < link (m) beyond touching -> drifts
    out, by_sp = [], {}
    for i, x, y, r in plan:
        by_sp.setdefault(i, []).append((x, y, r))
    for i, pts in by_sp.items():
        left = list(range(len(pts)))
        while left:
            stack, comp = [left.pop(0)], []
            while stack:
                a = stack.pop()
                comp.append(a)
                for b in left[:]:
                    if math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) \
                            < link + pts[a][2] + pts[b][2]:
                        left.remove(b)
                        stack.append(b)
            out.append((i, [pts[a] for a in comp]))
    return out


# ── v3 metrics (1 = clean) ──────────────────────────────────────────────────

def m_overlap3(plan):
    # crowding counts only within a stratum — groundcover under a shrub is design
    if len(plan) < 2:
        return 1.0
    bad = tot = 0
    for a in range(len(plan)):
        ia, xa, ya, ra = plan[a]
        for b in range(a + 1, len(plan)):
            ib, xb, yb, rb = plan[b]
            if not _conflict(ia, ib):
                continue
            tot += 1
            if math.hypot(xa - xb, ya - yb) < 0.75 * (ra + rb):
                bad += 1
    return 1 - bad / tot if tot else 1.0


def m_ground(plan, w, d):
    # "nature abhors bare soil": >=90% of the bed under some canopy
    n = hit = 0
    for a in range(40):
        for b in range(16):
            x, y = (a + 0.5) * w / 40, (b + 0.5) * d / 16
            n += 1
            if any((x - px) ** 2 + (y - py) ** 2 < pr * pr for _, px, py, pr in plan):
                hit += 1
    return min(1.0, (hit / n) / 0.9)


_SHARE_BANDS = {"structural": (0.02, 0.15), "seasonal": (0.2, 0.45),
                "filler": (0.0, 0.15), "groundcover": (0.35, 0.65)}


def m_layers(plan):
    # plant-count shares per layer vs the Rainer & West bands
    if not plan:
        return 0.0
    c = Counter(PALETTE[i]["layer"] for i, _, _, _ in plan)
    tot = len(plan)
    s = 0.0
    for lay, (lo, hi) in _SHARE_BANDS.items():
        f = c.get(lay, 0) / tot
        s += 1.0 if lo <= f <= hi else max(0.0, 1 - 6 * (lo - f if f < lo else f - hi))
    return s / len(_SHARE_BANDS)


def m_drift3(plan):
    # drifts should be elongated (aspect >= ~2.5) and oblique, not blobs/rows
    scores = []
    for i, pts in _clusters(plan):
        if len(pts) < 3 or PALETTE[i]["layer"] in ("structural", "groundcover"):
            continue
        xy = np.array([(x, y) for x, y, _ in pts])
        ev, vec = np.linalg.eigh(np.cov(xy.T) + np.eye(2) * 1e-9)
        aspect = math.sqrt(ev[1] / max(ev[0], 1e-9))
        ang = abs(math.degrees(math.atan2(vec[1, 1], vec[0, 1]))) % 180
        ang = min(ang, 180 - ang)
        s_el = min(1.0, aspect / 2.5)
        s_ob = 1.0 if 10 <= ang <= 80 else 0.5
        scores.append(s_el * s_ob)
    return float(np.mean(scores)) if scores else 0.0


def m_rhythm(plan, w):
    # some non-groundcover species repeats >=3x, spread wide, at even-ish gaps
    cl = {}
    for i, pts in _clusters(plan):
        if PALETTE[i]["layer"] not in ("structural", "groundcover"):
            cl.setdefault(i, []).append(float(np.mean([x for x, _, _ in pts])))
    best = 0.0
    for i, xs in cl.items():
        if len(xs) < 3:
            continue
        xs = sorted(xs)
        spread = min(1.0, (xs[-1] - xs[0]) / (0.6 * w))
        gaps = np.diff(xs)
        cv = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
        even = 1.0 if cv <= 0.4 else max(0.0, 1 - (cv - 0.4))
        best = max(best, 0.5 * spread + 0.5 * even)
    return best


def m_form(plan):
    # neighboring drifts should contrast in form or texture; coarse stays scarce
    cl = [(i, np.array([(x, y) for x, y, _ in pts]).mean(0))
          for i, pts in _clusters(plan) if PALETTE[i]["layer"] != "groundcover"]
    pairs = diff = 0
    for a in range(len(cl)):
        for b in range(a + 1, len(cl)):
            if np.hypot(*(cl[a][1] - cl[b][1])) < 1.8:
                pairs += 1
                pa, pb = PALETTE[cl[a][0]], PALETTE[cl[b][0]]
                if pa["form"] != pb["form"] or pa["tex"] != pb["tex"]:
                    diff += 1
    contrast = diff / pairs if pairs else 0.0
    area = {"fine": 0.0, "medium": 0.0, "coarse": 0.0}
    for i, _, _, r in plan:
        area[PALETTE[i]["tex"]] += r * r
    tot = sum(area.values()) or 1.0
    coarse_ok = 1.0 if area["coarse"] / tot <= 0.25 else max(0.0, 1 - 2 * (area["coarse"] / tot - 0.25))
    return 0.7 * contrast + 0.3 * coarse_ok


_WINDOWS = ((4, 5), (6, 7), (8, 9), (10, 11))


def m_season(plan, w):
    # min over 4 seasonal snapshots of how many bed thirds hold interest —
    # a border fails at its worst month; autumn counts structure persistence
    if not plan:
        return 0.0
    worst = 1.0
    for wi, (m0, m1) in enumerate(_WINDOWS):
        thirds = set()
        for i, x, _, _ in plan:
            p = PALETTE[i]
            b = p["bloom"]
            alive = (b[1] and b[0] <= m1 and b[1] >= m0) or (wi == 3 and p["persist"])
            if alive:
                thirds.add(min(2, int(3 * x / w)))
        worst = min(worst, len(thirds) / 3)
    return worst


def _hue(color):
    import colorsys

    from matplotlib.colors import to_rgb
    h, s, v = colorsys.rgb_to_hsv(*to_rgb(color))
    return h * 360, s


def m_harmony(plan):
    # Moon-Spencer-style: neighboring drifts that bloom together should sit in
    # identity/similarity (<=43 deg) or contrast (>=100 deg), not the ambiguous gap
    cl = []
    for i, pts in _clusters(plan):
        h, s = _hue(PALETTE[i]["color"])
        if s < 0.25 or not PALETTE[i]["bloom"][1]:      # neutrals sit out
            continue
        cl.append((i, np.array([(x, y) for x, y, _ in pts]).mean(0), h))
    pairs = good = 0
    for a in range(len(cl)):
        for b in range(a + 1, len(cl)):
            ba, bb = PALETTE[cl[a][0]]["bloom"], PALETTE[cl[b][0]]["bloom"]
            if ba[0] > bb[1] or bb[0] > ba[1]:          # never bloom together
                continue
            if np.hypot(*(cl[a][1] - cl[b][1])) >= 1.8:
                continue
            pairs += 1
            dh = abs(cl[a][2] - cl[b][2])
            dh = min(dh, 360 - dh)
            if dh <= 43 or dh >= 100:
                good += 1
    return good / pairs if pairs else 1.0


def m_cluster(plan):
    # Ripley-flavored: each repeated species should be clustered at drift scale
    by_sp = {}
    for i, x, y, r in plan:
        if PALETTE[i]["layer"] not in ("structural", "groundcover"):
            by_sp.setdefault(i, []).append((x, y, r))
    ok = n = 0
    for i, pts in by_sp.items():
        if len(pts) < 3:
            continue
        n += 1
        nn = [min(math.hypot(x - x2, y - y2) for k2, (x2, y2, _) in enumerate(pts) if k2 != k)
              for k, (x, y, _) in enumerate(pts)]
        if float(np.mean(nn)) <= 2.6 * pts[0][2] + 0.3:
            ok += 1
    return ok / n if n else 0.0


def score3(plan, w, d, sun=None):
    m = {"overlap": m_overlap3(plan), "ground": m_ground(plan, w, d),
         "layers": m_layers(plan), "drift": m_drift3(plan),
         "rhythm": m_rhythm(plan, w), "form": m_form(plan),
         "season": m_season(plan, w), "harmony": m_harmony(plan),
         "cluster": m_cluster(plan)}
    if sun is not None:
        m["sun"] = m_sun(plan, sun)
    m["score"] = float(np.mean(list(m.values())))
    return m


def repair3(plan, w, d):
    # like repair(), but only same-stratum pairs push apart — groundcover stays
    # put under taller layers; even-count drop skips carpets
    plan = [list(p) for p in plan]
    for _ in range(40):
        moved = False
        for a in range(len(plan)):
            for b in range(a + 1, len(plan)):
                if not _conflict(plan[a][0], plan[b][0]):
                    continue
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
    counts = Counter(p[0] for p in plan
                     if PALETTE[p[0]]["layer"] in ("seasonal", "filler"))
    for sp, cnt in counts.items():
        if cnt >= 2 and cnt % 2 == 0:
            k = max(i for i, p in enumerate(plan) if p[0] == sp)
            del plan[k]
    return [tuple(p) for p in plan]


# ── drawing ─────────────────────────────────────────────────────────────────

def show_plan(plan, w, d, ax, title="", pins=()):
    ax.set_facecolor("linen")
    ax.add_patch(Rectangle((0, 0), w, d, facecolor="wheat", edgecolor="sienna"))
    # short plants first so taller canopies draw over their groundcover
    for k in sorted(range(len(plan)), key=lambda j: PALETTE[plan[j][0]]["h"]):
        i, x, y, r = plan[k]
        pinned = k in pins
        gc = PALETTE[i]["layer"] == "groundcover"    # mats are foliage most of the year
        ax.add_patch(Circle((x, y), r, facecolor=PALETTE[i]["color"],
                            edgecolor="black" if pinned else "sienna",
                            lw=1.8 if pinned else 0.6, alpha=0.35 if gc else 0.85))
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
