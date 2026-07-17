"""Request/response models shared across the API.

Responses are mostly returned as plain dicts for flexibility; these models
document the shapes the frontend's TypeScript types mirror.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class StatusUpdate(BaseModel):
    status: str  # "planned" | "training" | "live"


class ArrangeRequest(BaseModel):
    aspect: str = "sun"           # "sun" | "part" | "shade"
    zone: str = "6a"
    area: float = 2500.0          # bed area in svg units (proxy for real m^2)
    toolbox: list[str] = []       # species names the user has favorited


class IllustrateRequest(BaseModel):
    species: str
    style: str = "Copperplate engraving"
    seed: Optional[int] = None


class RenderRequest(BaseModel):
    style: str = "Copperplate engraving"
    as_plate: bool = True
    beds: list[dict] = []
