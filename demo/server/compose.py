"""Compose engine: serve garden plans from the trained masked-diffusion layout model
(notebook 11, checkpoints/garden_maskdiff_best.pt), falling back to the rule generator.

Loads bvtrain/garden.py directly by file path (not the bvtrain package) so the demo
stays decoupled from the training stack's imports. Torch is imported lazily, mirroring
classifier.py — without torch or the checkpoint, Compose degrades to rules or a stub.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "garden_maskdiff_best.pt"

# token scheme — must mirror notebooks/11_complete_layout.ipynb exactly
NX, NY, NW, ND = 32, 12, 9, 7
MASK = 0
T_W = 1
T_D = T_W + NW
T_SUN = T_D + ND
T_SP = T_SUN + 3
T_X = T_SP + 1 + 16
T_Y = T_X + 1 + NX
VOCAB = T_Y + 1 + NY
MAXP = 60
L = 3 + 3 * MAXP

_garden = None
_model = None
_model_tried = False
_fam = None


def garden():
    global _garden
    if _garden is None:
        spec = importlib.util.spec_from_file_location("bv_garden", ROOT / "bvtrain" / "garden.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _garden = mod
    return _garden


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _load_model():
    global _model, _model_tried
    if _model is not None or _model_tried:
        return _model
    _model_tried = True
    torch = _torch()
    if torch is None or not CKPT.exists():
        return None
    try:
        import torch.nn as nn

        class LayoutMDM(nn.Module):
            def __init__(self, v=VOCAB, dm=192, heads=6, layers=6, ctx=L):
                super().__init__()
                self.tok = nn.Embedding(v, dm)
                self.pos = nn.Embedding(ctx, dm)
                blk = nn.TransformerEncoderLayer(dm, heads, dm * 4, dropout=0.1,
                                                 batch_first=True, norm_first=True)
                self.tf = nn.TransformerEncoder(blk, layers, enable_nested_tensor=False)
                self.head = nn.Linear(dm, v)

            def forward(self, x):
                h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
                return self.head(self.tf(h))

        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        m = LayoutMDM()
        m.load_state_dict(ck["model"])
        m.eval()
        _model = m
    except Exception:
        _model = None
    return _model


def status() -> str:
    if _load_model() is not None:
        return "live"
    try:
        garden()
        return "partial"      # rules only
    except Exception:
        return "prototype"


def palette() -> list[dict]:
    return [dict(p, idx=i) for i, p in enumerate(garden().PALETTE)]


def _site_tokens(w, d, sun):
    return [T_W + min(NW - 1, int((w - 3.5) / 0.5)),
            T_D + min(ND - 1, int((d - 1.8) / 0.2)), T_SUN + sun]


def _fam_mask(torch):
    global _fam
    if _fam is None:
        fam = [(T_W, T_D), (T_D, T_SUN), (T_SUN, T_SP)] + \
              [(T_SP, T_X), (T_X, T_Y), (T_Y, VOCAB)] * MAXP
        m = torch.full((L, VOCAB), float("-inf"))
        for p, (lo, hi) in enumerate(fam):
            m[p, lo:hi] = 0
        _fam = m
    return _fam


def _decode(tokens, w, d):
    g = garden()
    plan = []
    for k in range(3, L, 3):
        s, x, y = tokens[k] - T_SP, tokens[k + 1] - T_X, tokens[k + 2] - T_Y
        if s <= 0 or x <= 0 or y <= 0:
            continue
        i = s - 1
        plan.append((i, (x - 0.5) / NX * w, (y - 0.5) / NY * d, g.PALETTE[i]["s"] / 200))
    return plan


def _sample(model, w, d, sun, pin_slots, steps=12):
    torch = _torch()
    import torch.nn.functional as F
    canvas = torch.zeros(1, L, dtype=torch.long)
    fixed = torch.zeros(1, L, dtype=torch.bool)
    nonone = torch.zeros(1, L, dtype=torch.bool)
    canvas[0, :3] = torch.tensor(_site_tokens(w, d, sun))
    fixed[0, :3] = True
    for slot, sp_idx in pin_slots.items():
        p = 3 + slot * 3
        canvas[0, p] = T_SP + 1 + sp_idx
        fixed[0, p] = True
        nonone[0, p + 1] = nonone[0, p + 2] = True
    canvas[~fixed] = MASK
    m0 = int((canvas == MASK).sum())
    with torch.no_grad():
        for t in range(steps):
            logits = model(canvas) + _fam_mask(torch)
            logits[..., T_X] = logits[..., T_X].masked_fill(nonone, float("-inf"))
            logits[..., T_Y] = logits[..., T_Y].masked_fill(nonone, float("-inf"))
            probs = F.softmax(logits, -1)
            samp = torch.multinomial(probs.reshape(-1, VOCAB), 1).reshape(1, L)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)
            still = canvas == MASK
            canvas = torch.where(still, samp, canvas)
            n_mask = int(m0 * math.cos(math.pi / 2 * (t + 1) / steps))
            if n_mask:
                conf = conf.masked_fill(~still, float("inf"))
                canvas[0, conf[0].argsort()[:n_mask]] = MASK
    return _decode(canvas[0].tolist(), w, d)


def generate(width: float, depth: float, sun: int, pins: list[dict]) -> dict:
    g = garden()
    w = max(3.5, min(7.5, float(width)))
    d = max(1.8, min(3.0, float(depth)))
    sun = max(0, min(2, int(sun)))

    by_name = {p["name"].lower(): i for i, p in enumerate(g.PALETTE)}
    resolved, ignored = [], []
    for pin in pins or []:
        i = by_name.get(str(pin.get("species", "")).lower())
        if i is None:
            ignored.append(pin.get("species"))
        else:
            resolved += [i] * max(1, min(9, int(pin.get("count", 1))))
    resolved = resolved[:MAXP // 2]

    model = _load_model()
    if model is not None:
        # back-to-front canvas order: taller pinned species get earlier slots
        resolved.sort(key=lambda i: -g.PALETTE[i]["h"])
        plan = _sample(model, w, d, sun, dict(enumerate(resolved)))
        served, note = "diffusion", None
    else:
        plan = g.gen_plan(w, d, sun)
        served = "rules"
        note = "layout model checkpoint not found — serving the rule generator; pins not placed"

    pinned = {g.PALETTE[i]["name"] for i in resolved} if served == "diffusion" else set()
    plants = [{"species": g.PALETTE[i]["name"], "layer": g.PALETTE[i]["layer"],
               "color": g.PALETTE[i]["color"], "h": g.PALETTE[i]["h"],
               "x": round(x, 3), "y": round(y, 3), "r": round(r, 3),
               "pinned": g.PALETTE[i]["name"] in pinned}
              for i, x, y, r in plan]
    metrics = {k: round(v, 3) for k, v in g.score(plan, w, d, sun).items()}
    out = {"live": True, "served": served,
           "bed": {"w": w, "d": d, "sun": sun, "sun_name": g.SUN_NAMES[sun]},
           "plants": plants, "metrics": metrics, "ignored_pins": ignored}
    if note:
        out["note"] = note
    return out
