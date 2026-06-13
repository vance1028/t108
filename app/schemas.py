from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class BehaviorInput(BaseModel):
    user_id: str
    item_id: str
    action: str
    timestamp: Optional[float] = 0.0


class ItemInput(BaseModel):
    item_id: str
    tags: List[str] = Field(default_factory=list)


class TrainRequest(BaseModel):
    behaviors: List[BehaviorInput]
    items: Optional[List[ItemInput]] = Field(default_factory=list)


class RecommendResponseItem(BaseModel):
    item_id: str
    score: float
    reason: str


class SimilarItemResponseItem(BaseModel):
    item_id: str
    similarity: float


class TrainResponse(BaseModel):
    status: str
    num_users: int
    num_items: int
    num_behaviors: int
