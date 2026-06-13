from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.algorithms.item_cf import (
    ItemCFRecommender,
    UserBehavior,
    ItemProfile,
)
from app.schemas import (
    BehaviorInput,
    ItemInput,
    TrainRequest,
    TrainResponse,
    RecommendResponseItem,
    SimilarItemResponseItem,
)
from app.storage import ModelStorage


DATA_DIR = os.environ.get("RECOMMENDER_DATA_DIR", "data")
storage = ModelStorage(data_dir=DATA_DIR)
recommender: ItemCFRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = storage.load_model()
    if recommender is None:
        recommender = ItemCFRecommender()
    yield
    if recommender is not None and recommender.is_trained:
        storage.save_model(recommender)


app = FastAPI(
    title="Offline News Recommender",
    description="Item-based Collaborative Filtering Recommender Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    global recommender
    behaviors = [
        UserBehavior(
            user_id=b.user_id,
            item_id=b.item_id,
            action=b.action,
            timestamp=b.timestamp or 0.0,
        )
        for b in request.behaviors
    ]
    items = [
        ItemProfile(item_id=it.item_id, tags=it.tags)
        for it in (request.items or [])
    ]

    new_recommender = ItemCFRecommender()
    new_recommender.train(behaviors, items)

    storage.save_behaviors(behaviors)
    storage.save_items(items)
    storage.save_model(new_recommender)

    recommender = new_recommender

    return TrainResponse(
        status="ok",
        num_users=len(new_recommender.user_to_idx),
        num_items=len(new_recommender.item_to_idx),
        num_behaviors=len(behaviors),
    )


@app.get("/recommend", response_model=list[RecommendResponseItem])
async def recommend(
    user_id: str = Query(..., description="User ID"),
    top_n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
):
    global recommender
    if recommender is None or not recommender.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")

    results = recommender.recommend(user_id=user_id, top_n=top_n)
    return [
        RecommendResponseItem(item_id=r.item_id, score=r.score, reason=r.reason)
        for r in results
    ]


@app.get("/similar", response_model=list[SimilarItemResponseItem])
async def get_similar_items(
    item_id: str = Query(..., description="Item ID"),
    top_n: int = Query(10, ge=1, le=100, description="Number of similar items"),
):
    global recommender
    if recommender is None or not recommender.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")

    results = recommender.get_similar_items(item_id=item_id, top_n=top_n)
    return [
        SimilarItemResponseItem(item_id=r.item_id, similarity=r.similarity)
        for r in results
    ]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_trained": recommender is not None and recommender.is_trained,
    }
