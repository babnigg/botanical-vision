"""Request models for the demo API. Responses are returned as plain dicts."""
from __future__ import annotations

from pydantic import BaseModel


class ArrangeRequest(BaseModel):
    aspect: str = "sun"           # "sun" | "part" | "shade"
    zone: str = "6a"
    area: float = 2500.0          # bed area in svg units (proxy for real m^2)
    toolbox: list[str] = []       # species names the user has favorited


class ComposePin(BaseModel):
    species: str                  # a palette binomial, e.g. "Echinacea purpurea"
    count: int = 1


class ComposeRequest(BaseModel):
    width: float = 5.5            # bed width (m), clamped to 3.5-7.5
    depth: float = 2.6            # bed depth (m), clamped to 1.8-3.0
    sun: int = 2                  # 0 shade / 1 part / 2 full
    pins: list[ComposePin] = []   # toolbox species to lock into the plan


class SelectRequest(BaseModel):
    id: str                       # a model id from /api/models ("local:..." or "shared:...")
