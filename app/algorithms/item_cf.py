from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


ACTION_WEIGHTS = {
    "view": 1.0,
    "like": 3.0,
    "collect": 5.0,
}


@dataclass
class UserBehavior:
    user_id: str
    item_id: str
    action: str
    timestamp: float = 0.0


@dataclass
class ItemProfile:
    item_id: str
    tags: List[str] = field(default_factory=list)


@dataclass
class RecommendResult:
    item_id: str
    score: float
    reason: str


@dataclass
class SimilarItemResult:
    item_id: str
    similarity: float


class ItemCFRecommender:
    def __init__(
        self,
        action_weights: Optional[Dict[str, float]] = None,
        similarity_threshold: float = 0.0,
    ):
        self.action_weights = action_weights or ACTION_WEIGHTS
        self.similarity_threshold = similarity_threshold

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.user_item_matrix: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None

        self.item_popularity: Dict[str, float] = defaultdict(float)
        self.item_profiles: Dict[str, ItemProfile] = {}
        self.tag_items: Dict[str, List[str]] = defaultdict(list)

        self.is_trained: bool = False

    def _build_index(self, behaviors: List[UserBehavior], items: List[ItemProfile]) -> None:
        user_ids = sorted({b.user_id for b in behaviors})
        item_ids_set = {b.item_id for b in behaviors}
        for it in items:
            item_ids_set.add(it.item_id)
        item_ids = sorted(item_ids_set)

        self.user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}

    def _build_user_item_matrix(self, behaviors: List[UserBehavior]) -> None:
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        matrix = np.zeros((n_users, n_items), dtype=np.float64)

        for b in behaviors:
            weight = self.action_weights.get(b.action, 1.0)
            u_idx = self.user_to_idx[b.user_id]
            i_idx = self.item_to_idx[b.item_id]
            matrix[u_idx, i_idx] += weight
            self.item_popularity[b.item_id] += weight

        self.user_item_matrix = matrix

    def _build_item_profiles(self, items: List[ItemProfile]) -> None:
        self.item_profiles = {}
        self.tag_items = defaultdict(list)
        for it in items:
            self.item_profiles[it.item_id] = it
            for tag in it.tags:
                self.tag_items[tag].append(it.item_id)

    def _compute_adjusted_cosine_similarity(self) -> None:
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not built")

        n_users, n_items = self.user_item_matrix.shape
        user_means = self.user_item_matrix.mean(axis=1, keepdims=True)
        user_means = np.where(user_means == 0, 1e-9, user_means)
        adjusted = self.user_item_matrix - user_means

        norms = np.linalg.norm(adjusted, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normalized = adjusted / norms

        similarity = normalized.T @ normalized
        np.fill_diagonal(similarity, 0.0)
        similarity = np.clip(similarity, -1.0, 1.0)

        self.item_similarity = similarity

    def _compute_tag_similarity(self, item_a: str, item_b: str) -> float:
        profile_a = self.item_profiles.get(item_a)
        profile_b = self.item_profiles.get(item_b)
        if not profile_a or not profile_b:
            return 0.0
        tags_a = set(profile_a.tags)
        tags_b = set(profile_b.tags)
        if not tags_a or not tags_b:
            return 0.0
        intersection = len(tags_a & tags_b)
        union = len(tags_a | tags_b)
        return intersection / union if union > 0 else 0.0

    def train(
        self,
        behaviors: List[UserBehavior],
        items: Optional[List[ItemProfile]] = None,
    ) -> None:
        items = items or []
        self._build_index(behaviors, items)
        self._build_user_item_matrix(behaviors)
        self._build_item_profiles(items)
        self._compute_adjusted_cosine_similarity()
        self.is_trained = True

    def _get_cold_start_items(self, user_id: str, top_n: int) -> List[RecommendResult]:
        if self.item_popularity:
            sorted_items = sorted(
                self.item_popularity.items(), key=lambda x: x[1], reverse=True
            )
            return [
                RecommendResult(item_id=iid, score=score, reason="热门推荐")
                for iid, score in sorted_items[:top_n]
            ]
        if self.item_profiles:
            return [
                RecommendResult(item_id=iid, score=1.0, reason="新用户/新内容推荐")
                for iid in list(self.item_profiles.keys())[:top_n]
            ]
        return []

    def _get_tag_based_items(
        self, user_items: List[str], exclude_items: set, top_n: int
    ) -> List[RecommendResult]:
        tag_scores: Dict[str, float] = defaultdict(float)
        for iid in user_items:
            profile = self.item_profiles.get(iid)
            if profile:
                for tag in profile.tags:
                    tag_scores[tag] += 1.0

        candidate_scores: Dict[str, float] = defaultdict(float)
        for tag, tscore in tag_scores.items():
            for iid in self.tag_items.get(tag, []):
                if iid in exclude_items:
                    continue
                candidate_scores[iid] += tscore

        if not candidate_scores:
            return []

        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [
            RecommendResult(
                item_id=iid,
                score=score,
                reason="基于内容标签推荐",
            )
            for iid, score in sorted_candidates[:top_n]
        ]

    def recommend(
        self, user_id: str, top_n: int = 10, min_similar_items: int = 5
    ) -> List[RecommendResult]:
        if not self.is_trained:
            raise ValueError("Model not trained, call train() first")

        if user_id not in self.user_to_idx:
            return self._get_cold_start_items(user_id, top_n)

        u_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix[u_idx]
        interacted_indices = np.where(user_vector > 0)[0]

        if len(interacted_indices) == 0:
            return self._get_cold_start_items(user_id, top_n)

        interacted_items = [self.idx_to_item[i] for i in interacted_indices]
        interacted_set = set(interacted_items)

        scores: Dict[str, float] = defaultdict(float)
        reasons: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        for i_idx in interacted_indices:
            i_iid = self.idx_to_item[i_idx]
            weight = user_vector[i_idx]
            sim_vector = self.item_similarity[i_idx]

            for j_idx in range(len(sim_vector)):
                sim = sim_vector[j_idx]
                if sim <= self.similarity_threshold:
                    continue
                j_iid = self.idx_to_item[j_idx]
                if j_iid in interacted_set:
                    continue
                contribution = sim * weight
                scores[j_iid] += contribution
                reasons[j_iid].append((i_iid, sim))

        if not scores:
            tag_based = self._get_tag_based_items(
                interacted_items, interacted_set, top_n
            )
            if tag_based:
                return tag_based
            return self._get_cold_start_items(user_id, top_n)

        def make_reason(item_id: str) -> str:
            top_reasons = sorted(reasons[item_id], key=lambda x: x[1], reverse=True)[:2]
            reason_items = [r[0] for r in top_reasons]
            if len(reason_items) == 1:
                return f"因为你读过 {reason_items[0]}"
            return f"因为你读过 {reason_items[0]} 和 {reason_items[1]}"

        results = []
        for iid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            results.append(
                RecommendResult(item_id=iid, score=score, reason=make_reason(iid))
            )
            if len(results) >= top_n:
                break

        if len(results) < top_n:
            tag_based = self._get_tag_based_items(
                interacted_items, interacted_set | {r.item_id for r in results},
                top_n - len(results),
            )
            results.extend(tag_based)

        return results[:top_n]

    def get_similar_items(
        self, item_id: str, top_n: int = 10
    ) -> List[SimilarItemResult]:
        if not self.is_trained:
            raise ValueError("Model not trained, call train() first")

        if item_id not in self.item_to_idx:
            if item_id in self.item_profiles:
                return self._get_similar_by_tags(item_id, top_n)
            return []

        i_idx = self.item_to_idx[item_id]
        sim_vector = self.item_similarity[i_idx]

        results = []
        for j_idx in range(len(sim_vector)):
            j_iid = self.idx_to_item[j_idx]
            if j_iid == item_id:
                continue
            sim = sim_vector[j_idx]
            if sim <= self.similarity_threshold:
                continue
            results.append(SimilarItemResult(item_id=j_iid, similarity=float(sim)))

        results.sort(key=lambda x: x.similarity, reverse=True)

        if len(results) < top_n and item_id in self.item_profiles:
            tag_similar = self._get_similar_by_tags(item_id, top_n)
            existing = {r.item_id for r in results}
            for ts in tag_similar:
                if ts.item_id not in existing:
                    results.append(ts)
                    if len(results) >= top_n:
                        break

        return results[:top_n]

    def _get_similar_by_tags(
        self, item_id: str, top_n: int
    ) -> List[SimilarItemResult]:
        results = []
        for other_id in self.item_profiles:
            if other_id == item_id:
                continue
            sim = self._compute_tag_similarity(item_id, other_id)
            if sim > 0:
                results.append(SimilarItemResult(item_id=other_id, similarity=sim))
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_n]

    def get_state(self) -> Dict:
        return {
            "action_weights": self.action_weights,
            "similarity_threshold": self.similarity_threshold,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": {int(k): v for k, v in self.idx_to_user.items()},
            "item_to_idx": self.item_to_idx,
            "idx_to_item": {int(k): v for k, v in self.idx_to_item.items()},
            "user_item_matrix": self.user_item_matrix.tolist() if self.user_item_matrix is not None else None,
            "item_similarity": self.item_similarity.tolist() if self.item_similarity is not None else None,
            "item_popularity": dict(self.item_popularity),
            "item_profiles": {
                k: {"item_id": v.item_id, "tags": v.tags}
                for k, v in self.item_profiles.items()
            },
            "tag_items": {k: list(v) for k, v in self.tag_items.items()},
            "is_trained": self.is_trained,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "ItemCFRecommender":
        rec = cls(
            action_weights=state.get("action_weights"),
            similarity_threshold=state.get("similarity_threshold", 0.0),
        )
        rec.user_to_idx = state["user_to_idx"]
        rec.idx_to_user = {int(k): v for k, v in state["idx_to_user"].items()}
        rec.item_to_idx = state["item_to_idx"]
        rec.idx_to_item = {int(k): v for k, v in state["idx_to_item"].items()}

        uim = state.get("user_item_matrix")
        rec.user_item_matrix = np.array(uim, dtype=np.float64) if uim is not None else None

        sim = state.get("item_similarity")
        rec.item_similarity = np.array(sim, dtype=np.float64) if sim is not None else None

        rec.item_popularity = defaultdict(float, state.get("item_popularity", {}))
        rec.item_profiles = {
            k: ItemProfile(item_id=v["item_id"], tags=v.get("tags", []))
            for k, v in state.get("item_profiles", {}).items()
        }
        rec.tag_items = defaultdict(
            list, {k: list(v) for k, v in state.get("tag_items", {}).items()}
        )
        rec.is_trained = state.get("is_trained", False)
        return rec
