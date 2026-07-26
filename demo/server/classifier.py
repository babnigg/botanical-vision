"""Classifier serving for the demo — you pick which model to serve (no auto-pick).

The active model is chosen via /api/models/select and persisted to a small file next to
the checkpoints, so it survives restarts. It can be a local checkpoint (checkpoints/*.pt)
or one pulled from the shared Hugging Face repo — either way a {state_dict, labels, ...}
bundle. We rebuild ResNet-50 + the eval transform and return top-5 {species, confidence}.
If torch is missing or nothing is selected, a labeled stub keeps the frontend alive.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from .traits import taxonomy

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
SEL_FILE = CKPT_DIR / ".demo_selected.json"
SEG_SEL_FILE = CKPT_DIR / ".demo_seg_selected.json"

# ImageNet normalization + 224 crop — must match the notebooks (03/04/05/06).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
TOP_K = 5

# Gaussian-blur radius for the soft-mask background blend. Hard-masking to black
# would shove the classifier's input off the training distribution and *hurt*
# top-1; blending against a strong blur preserves colors/textures.
BG_BLUR_RADIUS = 15

_STUB = {
    "coneflower": [("Echinacea purpurea", 62.4), ("Echinacea pallida", 18.1),
                   ("Rudbeckia hirta", 9.7), ("Ratibida pinnata", 4.3),
                   ("Echinacea angustifolia", 2.1)],
    "iris": [("Iris versicolor", 47.8), ("Iris virginica", 24.9),
             ("Iris setosa", 12.1), ("Sisyrinchium montanum", 6.0),
             ("Iris lacustris", 3.4)],
}

_predictor = None
_loaded_id: str | None = None
_active: dict | None = None

_segmenter = None
_seg_loaded_path: str | None = None


def _device():
    """CUDA if available, else CPU. Both the classifier and the segmenter run here."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def active_id() -> str | None:
    """The currently-selected model id (persisted), or None if nothing is chosen yet."""
    if SEL_FILE.exists():
        try:
            return json.loads(SEL_FILE.read_text(encoding="utf-8")).get("id")
        except Exception:
            return None
    return None


def set_active(model_id: str) -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    SEL_FILE.write_text(json.dumps({"id": model_id}), encoding="utf-8")
    global _predictor, _loaded_id, _active
    _predictor, _loaded_id, _active = None, None, None   # force a reload on next use


def clear_active() -> None:
    global _predictor, _loaded_id, _active
    _predictor, _loaded_id, _active = None, None, None
    try:
        SEL_FILE.unlink()
    except Exception:
        pass


def _resolve_path(model_id: str) -> Path | None:
    if model_id.startswith("local:"):
        p = CKPT_DIR / model_id.split(":", 1)[1]
        return p if p.exists() else None
    if model_id.startswith("shared:"):
        ref = model_id.split(":", 1)[1]
        if "/" not in ref:
            return None
        author, name = ref.split("/", 1)
        try:
            from .models import hf_pull
            return Path(hf_pull(author, name))
        except Exception:
            return None
    return None


def _build(path: Path):
    import torch
    from torch import nn
    from torchvision import models, transforms

    b = torch.load(str(path), map_location="cpu", weights_only=False)
    labels = b["labels"]
    net = models.resnet50()
    net.fc = nn.Linear(net.fc.in_features, len(labels))
    net.load_state_dict(b["state_dict"])
    net.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(b.get("img_size", IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(b.get("mean", MEAN), b.get("std", STD)),
    ])

    def predict_topk(img):
        x = tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            prob = net(x).float().softmax(1)[0]
        vals, idx = prob.topk(TOP_K)
        return [(labels[int(i)], float(v) * 100) for v, i in zip(vals, idx)]

    return predict_topk, labels


def _load():
    """Lazily load + cache the predictor for the currently-selected model."""
    global _predictor, _loaded_id, _active
    mid = active_id()
    if mid is None:
        _predictor, _loaded_id, _active = None, None, None
        return None
    if _predictor is not None and _loaded_id == mid:
        return _predictor
    if not torch_available():
        return None
    path = _resolve_path(mid)
    if path is None:
        _predictor, _active = None, None
        return None
    try:
        pred, labels = _build(path)
    except Exception:
        _predictor, _active = None, None
        return None
    name = mid.split(":", 1)[1]
    if mid.startswith("local:"):
        name = Path(name).stem
    _predictor, _loaded_id = pred, mid
    _active = {"id": mid, "source": mid.split(":", 1)[0], "name": name, "species": len(labels)}
    return _predictor


def served_model() -> dict | None:
    """{id, source, name, species} of the model actually loaded, or None."""
    _load()
    return _active


def _resolve_seg_path() -> Path | None:
    """Which segmenter checkpoint to serve. Explicit selection wins; otherwise
    the newest local ``checkpoints/*_seg_best.pt`` (if any). Missing = segmentation
    stage is a no-op."""
    if SEG_SEL_FILE.exists():
        try:
            p = Path(json.loads(SEG_SEL_FILE.read_text(encoding="utf-8")).get("path", ""))
            if p.exists():
                return p
        except Exception:
            pass
    hits = sorted(CKPT_DIR.glob("*_seg_best.pt"),
                  key=lambda x: x.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _build_seg(path: Path):
    """Load a segmentation checkpoint into a callable ``pil -> HxW plant-probability tensor``."""
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    from torchvision.transforms import functional as TF

    dev = _device()
    b = torch.load(str(path), map_location="cpu", weights_only=False)
    net = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=False)
    in_ch = net.classifier[-1].in_channels
    from torch import nn as _nn
    net.classifier[-1] = _nn.Conv2d(in_ch, 1, kernel_size=1)
    net.load_state_dict(b["state_dict"])
    net.to(dev).eval()

    def segment(pil_img):
        x = TF.resize(pil_img.convert("RGB"), 256)
        x = TF.center_crop(x, IMG_SIZE)
        x = TF.to_tensor(x)
        x = TF.normalize(x, MEAN, STD).unsqueeze(0).to(dev)
        with torch.no_grad():
            probs = torch.sigmoid(net(x)["out"])[0, 0].cpu()   # 224x224 float in [0, 1]
        return probs

    return segment


def _load_segmenter():
    """Lazily load + cache the segmenter, if any is available."""
    global _segmenter, _seg_loaded_path
    if not torch_available():
        return None
    path = _resolve_seg_path()
    if path is None:
        _segmenter, _seg_loaded_path = None, None
        return None
    if _segmenter is not None and str(path) == _seg_loaded_path:
        return _segmenter
    try:
        seg = _build_seg(path)
    except Exception:
        _segmenter, _seg_loaded_path = None, None
        return None
    _segmenter, _seg_loaded_path = seg, str(path)
    return _segmenter


def _mask_to_original(prob_mask, w: int, h: int, resize_short: int = 256,
                       crop_size: int = IMG_SIZE):
    """Map the segmenter's 224x224 output back into original pixel coords.

    The segmenter (and classifier) apply Resize(``resize_short``, aspect-preserving) +
    CenterCrop(``crop_size``), so the 224x224 mask corresponds to a centered
    ``(crop_size / scale)`` square of the original, where ``scale = resize_short /
    min(w, h)``. We upsample the mask to that square and paste it into a full-size
    zeroed alpha — regions outside the segmenter's view stay at 0 (background).
    """
    import numpy as np
    from PIL import Image

    scale = resize_short / min(w, h)
    side = min(int(round(crop_size / scale)), w, h)
    arr = (prob_mask.numpy() * 255).astype(np.uint8)
    patch = Image.fromarray(arr, mode="L").resize((side, side), Image.BILINEAR)
    alpha = Image.new("L", (w, h), 0)
    alpha.paste(patch, ((w - side) // 2, (h - side) // 2))
    return alpha


def _apply_soft_mask(pil_img, alpha):
    """Blend the input with a Gaussian-blurred copy in the background regions."""
    from PIL import Image, ImageFilter

    fg = pil_img.convert("RGB")
    bg = fg.filter(ImageFilter.GaussianBlur(radius=BG_BLUR_RADIUS))
    return Image.composite(fg, bg, alpha)


def _mask_png_b64(alpha, max_side: int = 256) -> str:
    """Return the mask as a base64 PNG data URL for the frontend overlay.

    Encoded RGB with the mask in the G channel (R=B=0), so ``mix-blend-mode:
    multiply`` on the frontend darkens the background toward black and tints the
    plant green. A grayscale PNG can't be re-tinted via CSS filters on receive,
    so we bake the color here.
    """
    import base64
    import io as _io
    from PIL import Image

    w, h = alpha.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        alpha = alpha.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    zero = Image.new("L", alpha.size, 0)
    rgb = Image.merge("RGB", (zero, alpha, zero))
    buf = _io.BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def active_segmenter() -> dict | None:
    """{name, path} of the loaded segmenter, or None if segmentation is off."""
    _load_segmenter()
    if _seg_loaded_path is None:
        return None
    return {"name": Path(_seg_loaded_path).name, "path": _seg_loaded_path}


def _decorate(name: str, conf: float) -> dict:
    t = taxonomy(name)
    return {
        "species": name, "common": t["common"], "genus": t["genus"],
        "family": t["family"], "order": t["order"], "iucn": t["iucn"] or "LC",
        "confidence": round(conf, 2),
    }


def _agreement(top: list[dict]) -> dict:
    fam, gen = top[0]["family"], top[0]["genus"]
    return {
        "family": fam, "family_count": sum(1 for t in top if t["family"] == fam),
        "genus": gen, "genus_count": sum(1 for t in top if t["genus"] == gen),
    }


def _stub(sample: str | None, reason: str, note: str | None = None) -> dict:
    rows = _STUB.get(sample or "coneflower", _STUB["coneflower"])
    top = [_decorate(n, c) for n, c in rows]
    return {
        "live": False, "served": "stub", "reason": reason, "model": None,
        "predictions": top, "taxonomy_agreement": _agreement(top),
        "note": note or "Demo result — the classifier isn't running in this environment.",
    }


def predict(image_bytes: bytes | None, sample: str | None = None) -> dict:
    if not torch_available():
        return _stub(sample, "no_torch",
                     "PyTorch isn't installed here — demo result. "
                     "`pip install torch torchvision` to run the model on your image.")
    if active_id() is None:
        return _stub(sample, "no_model", "No model selected — pick one in the model picker above.")
    if _load() is None:
        return _stub(sample, "no_model", "The selected model couldn't be loaded — pick another.")
    if not image_bytes:
        return _stub(sample, "no_image")
    try:
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        seg = _load_segmenter()
        mask_b64 = None
        if seg is not None:
            probs = seg(img)                                        # 224x224 in [0, 1]
            alpha = _mask_to_original(probs, img.size[0], img.size[1])
            mask_b64 = _mask_png_b64(alpha)
            img = _apply_soft_mask(img, alpha)                      # classifier sees masked img
        rows = _predictor(img)
        top = [_decorate(n, c) for n, c in rows]
    except Exception:
        return _stub(sample, "infer_failed")
    return {
        "live": True, "served": "model", "reason": None,
        "model": (_active or {}).get("name"),
        "predictions": top, "taxonomy_agreement": _agreement(top), "note": None,
        "mask_png_b64": mask_b64,
        "segmenter": Path(_seg_loaded_path).name if _seg_loaded_path else None,
    }
