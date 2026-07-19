"""Request models for the demo API. Responses are returned as plain dicts."""
from __future__ import annotations

from pydantic import BaseModel


class ArrangeRequest(BaseModel):
    aspect: str = "sun"           # "sun" | "part" | "shade"
    zone: str = "6a"
    area: float = 2500.0          # bed area in svg units (proxy for real m^2)
    toolbox: list[str] = []       # species names the user has favorited


class SelectRequest(BaseModel):
    id: str                       # a model id from /api/models ("local:..." or "shared:...")
