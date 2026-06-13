from __future__ import annotations

import json
import os
from typing import List, Optional

from app.algorithms.item_cf import ItemCFRecommender, UserBehavior, ItemProfile


class ModelStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.model_path = os.path.join(data_dir, "model_state.json")
        self.behaviors_path = os.path.join(data_dir, "behaviors.json")
        self.items_path = os.path.join(data_dir, "items.json")
        os.makedirs(data_dir, exist_ok=True)

    def save_model(self, recommender: ItemCFRecommender) -> None:
        state = recommender.get_state()
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_model(self) -> Optional[ItemCFRecommender]:
        if not os.path.exists(self.model_path):
            return None
        with open(self.model_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return ItemCFRecommender.from_state(state)

    def save_behaviors(self, behaviors: List[UserBehavior]) -> None:
        data = [
            {
                "user_id": b.user_id,
                "item_id": b.item_id,
                "action": b.action,
                "timestamp": b.timestamp,
            }
            for b in behaviors
        ]
        with open(self.behaviors_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_behaviors(self) -> List[UserBehavior]:
        if not os.path.exists(self.behaviors_path):
            return []
        with open(self.behaviors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            UserBehavior(
                user_id=b["user_id"],
                item_id=b["item_id"],
                action=b["action"],
                timestamp=b.get("timestamp", 0.0),
            )
            for b in data
        ]

    def save_items(self, items: List[ItemProfile]) -> None:
        data = [
            {"item_id": it.item_id, "tags": it.tags} for it in items
        ]
        with open(self.items_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_items(self) -> List[ItemProfile]:
        if not os.path.exists(self.items_path):
            return []
        with open(self.items_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            ItemProfile(item_id=it["item_id"], tags=it.get("tags", []))
            for it in data
        ]

    def clear(self) -> None:
        for path in [self.model_path, self.behaviors_path, self.items_path]:
            if os.path.exists(path):
                os.remove(path)
