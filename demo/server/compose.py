"""Compose engine: garden plans from the iteration-5 masked-diffusion layout model
(notebook 16, checkpoints/garden_maskdiff4_best.pt). Decode-time selection is kept
honest: candidates are repaired, ranked by `realism` (distance to real-bed design
statistics — not our own rule metrics), and served randomly from the top 3. Falls
back to the curated rule generator without torch or the checkpoint.

Capability routing (from nb 16's blinded pairwise judgment): unpinned requests are
served by the curated rule generator, which humans preferred; pinned requests go to
the diffusion model, whose unique strength is infilling around fixed slot tokens
(guaranteed: >= requested plants appear, then trimmed back to the brief).

Loads bvtrain/garden.py directly by file path (not the bvtrain package) so the demo
stays decoupled from the training stack. Torch imports lazily, mirroring classifier.py.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import math
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "garden_maskdiff4_best.pt"

# token scheme — must mirror notebooks/16_curated_design.ipynb exactly
# (count tokens removed in iteration 5 — slot pins are the only pin mechanism)
NX, NY, NW, ND = 32, 12, 9, 7
NSP = 21
MASK = 0
T_W = 1
T_D = T_W + NW
T_SUN = T_D + ND
T_SP = T_SUN + 3
T_X = T_SP + 1 + NSP
T_Y = T_X + 1 + NX
VOCAB = T_Y + 1 + NY
MAXP = 48
L = 3 + 3 * MAXP

N_CANDIDATES = 24

_garden = None
_model = None
_model_tried = False
_fam = None
_render_jobs: dict = {}
_render_lock = threading.Lock()
_pipe = None


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
        return "partial"
    except Exception:
        return "prototype"


def palette() -> list[dict]:
    # slice to the model vocab — v3 palette additions aren't in this checkpoint
    return [dict(p, idx=i) for i, p in enumerate(garden().PALETTE[:NSP])]


def _site_tokens(w, d, sun):
    return [T_W + min(NW - 1, int((w - 3.5) / 0.5)),
            T_D + min(ND - 1, int((d - 1.8) / 0.2)), T_SUN + sun]


def _fam_mask(torch):
    global _fam
    if _fam is None:
        fam = ([(T_W, T_D), (T_D, T_SUN), (T_SUN, T_SP)]
               + [(T_SP, T_X), (T_X, T_Y), (T_Y, VOCAB)] * MAXP)
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


def _sample_batch(model, w, d, sun, pin_list, n, steps=12):
    torch = _torch()
    import torch.nn.functional as F
    canvas = torch.zeros(n, L, dtype=torch.long)
    fixed = torch.zeros(n, L, dtype=torch.bool)
    nonone = torch.zeros(n, L, dtype=torch.bool)
    site = torch.tensor(_site_tokens(w, d, sun))
    canvas[:, :3] = site
    fixed[:, :3] = True
    # slot pins guarantee presence
    slot = 0
    for sp_idx, cnt in pin_list:
        for _ in range(cnt):
            if slot >= MAXP:
                break
            p = 3 + slot * 3
            canvas[:, p] = T_SP + 1 + sp_idx
            fixed[:, p] = True
            nonone[:, p + 1] = nonone[:, p + 2] = True
            slot += 1
    canvas[~fixed] = MASK
    m0 = (canvas == MASK).sum(1)
    with torch.no_grad():
        for t in range(steps):
            logits = model(canvas) + _fam_mask(torch)
            logits[..., T_X] = logits[..., T_X].masked_fill(nonone, float("-inf"))
            logits[..., T_Y] = logits[..., T_Y].masked_fill(nonone, float("-inf"))
            probs = F.softmax(logits, -1)
            samp = torch.multinomial(probs.reshape(-1, VOCAB), 1).reshape(n, L)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)
            still = canvas == MASK
            canvas = torch.where(still, samp, canvas)
            n_mask = (m0.float() * math.cos(math.pi / 2 * (t + 1) / steps)).long()
            conf = conf.masked_fill(~still, float("inf"))
            for b in range(n):
                if n_mask[b] > 0:
                    canvas[b, conf[b].argsort()[:n_mask[b]]] = MASK
    return [_decode(canvas[b].tolist(), w, d) for b in range(n)]




def _trim_pinned(plan, pin_list):
    """the model over-serves pinned species (>= is guaranteed); trim whole masses
    beyond the brief so 7 pinned coneflowers don't arrive as 17."""
    g_ = garden()
    for sp_idx, cnt in pin_list:
        cl = sorted(([pt for pt in pts] for i, pts in g_._clusters(plan, link=0.35)
                     if i == sp_idx), key=len, reverse=True)
        keep, total = [], 0
        for pts in cl:
            if total >= cnt:
                break
            keep.append(pts)
            total += len(pts)
        kept = {(round(x, 6), round(y, 6)) for pts in keep for x, y, _ in pts}
        plan[:] = [p for p in plan
                   if p[0] != sp_idx or (round(p[1], 6), round(p[2], 6)) in kept]


def _patches(plan, pinned):
    """designer patches: one smooth outline per same-species mass (shapely union)."""
    g_ = garden()
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError:
        return []
    out = []
    for i, pts in g_._clusters(plan, link=0.35):
        p = g_.PALETTE[i]
        shapes = unary_union([Point(x, y).buffer(r) for x, y, r in pts]).buffer(0.13).buffer(-0.09)
        rings = [list(zip(*geom.exterior.xy)) for geom in getattr(shapes, "geoms", [shapes])]
        cx = sum(x for x, _, _ in pts) / len(pts)
        cy = sum(y for _, y, _ in pts) / len(pts)
        out.append({"species": p["name"], "common": p["common"], "layer": p["layer"],
                    "color": p["color"], "count": len(pts),
                    "pinned": p["name"] in pinned,
                    "label": [round(cx, 3), round(cy, 3)],
                    "rings": [[[round(x, 3), round(y, 3)] for x, y in ring] for ring in rings],
                    "crowns": [[round(x, 3), round(y, 3)] for x, y, _ in pts]})
    return out


def generate(width: float, depth: float, sun: int, pins: list[dict]) -> dict:
    g = garden()
    w = max(3.5, min(7.5, float(width)))
    d = max(1.8, min(3.0, float(depth)))
    sun = max(0, min(2, int(sun)))

    by_name = {p["name"].lower(): i for i, p in enumerate(g.PALETTE)}
    pin_list, ignored = [], []
    for pin in pins or []:
        i = by_name.get(str(pin.get("species", "")).lower())
        if i is None:
            ignored.append(pin.get("species"))
        else:
            pin_list.append((i, max(1, min(9, int(pin.get("count", 1))))))
    # taller pinned species take earlier (backer) slots
    pin_list.sort(key=lambda t: -g.PALETTE[t[0]]["h"])

    # capability routing (decided by blinded pairwise judgment, nb 16): the curated
    # rule generator composes nicer beds from scratch; the diffusion model's unique
    # value is infilling around pinned plants — so pins go to the model.
    model = _load_model() if pin_list else None
    import random as _random
    if model is not None:
        cands = [g.curate4(g.repair3(p, w, d), w, d)
                 for p in _sample_batch(model, w, d, sun, pin_list, N_CANDIDATES)]
        for c in cands:
            _trim_pinned(c, pin_list)
        ranked = sorted(cands, key=lambda p: g.realism(p, w, d))
        plan = _random.choice(ranked[:3])       # softened selection, no metric argmax
        served, note = f"diffusion4 best-of-{N_CANDIDATES}", None
    else:
        cands = [g.gen_plan4(w, d, sun) for _ in range(4)]
        plan = min(cands, key=lambda p: g.realism(p, w, d))
        served = "designer rules"
        note = None
        if pin_list and not CKPT.exists():
            note = "layout model checkpoint not found — pins not placed"

    pinned = {g.PALETTE[i]["name"] for i, _ in pin_list} if model is not None else set()
    plants = [{"species": g.PALETTE[i]["name"], "common": g.PALETTE[i]["common"],
               "layer": g.PALETTE[i]["layer"],
               "color": g.PALETTE[i]["color"], "h": g.PALETTE[i]["h"],
               "x": round(x, 3), "y": round(y, 3), "r": round(r, 3),
               "pinned": g.PALETTE[i]["name"] in pinned}
              for i, x, y, r in plan]
    metrics = {"realism": round(g.realism(plan, w, d), 2),
               "plants": len(plan), "species": len({i for i, *_ in plan})}
    out = {"live": True, "served": served,
           "bed": {"w": w, "d": d, "sun": sun, "sun_name": g.SUN_NAMES[sun]},
           "plants": plants, "patches": _patches(plan, pinned), "metrics": metrics,
           "ignored_pins": ignored}
    if note:
        out["note"] = note
    return out


# ── styled render jobs (sd 1.5 + seg controlnet; ~10-17 min on this machine) ──

def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    dtype = torch.float16 if vram >= 8 else torch.float32
    cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn,
        torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if vram and vram < 8:
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()
    else:
        pipe.to("cuda")
    _pipe = pipe
    return pipe


def _run_render(job_id, plants, w, d):
    import torch
    g = garden()
    job = _render_jobs[job_id]
    try:
        by_name = {p["name"]: i for i, p in enumerate(g.PALETTE)}
        plan = [(by_name[p["species"]], p["x"], p["y"], p["r"])
                for p in plants if p.get("species") in by_name]
        cond = g.seg_image(plan, w, d).resize((640, 384))
        job["status"] = "loading model"
        with _render_lock:                      # one render at a time
            pipe = _load_pipe()
            job["status"] = "rendering"
            t0 = time.time()
            img = pipe("top-down garden planting plan, perennial border with flowering "
                       "shrubs, lush, watercolor botanical illustration",
                       image=cond, num_inference_steps=20, guidance_scale=7.0,
                       controlnet_conditioning_scale=0.9,
                       generator=torch.Generator().manual_seed(0)).images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        job["png_b64"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        job["seconds"] = round(time.time() - t0, 1)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)[:200]


def start_render(plants: list[dict], w: float, d: float) -> dict:
    if _torch() is None:
        return {"ok": False, "error": "torch unavailable"}
    job_id = uuid.uuid4().hex[:12]
    _render_jobs[job_id] = {"status": "queued", "started": time.time()}
    threading.Thread(target=_run_render, args=(job_id, plants, w, d), daemon=True).start()
    return {"ok": True, "job": job_id}


def render_status(job_id: str) -> dict:
    job = _render_jobs.get(job_id)
    if job is None:
        return {"status": "unknown"}
    out = {"status": job["status"], "elapsed": round(time.time() - job["started"], 1)}
    if job["status"] == "done":
        out["png_b64"] = job["png_b64"]
        out["seconds"] = job["seconds"]
    if job.get("error"):
        out["error"] = job["error"]
    return out
