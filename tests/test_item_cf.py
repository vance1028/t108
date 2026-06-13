import os
import sys
import shutil
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.algorithms.item_cf import (
    ItemCFRecommender,
    UserBehavior,
    ItemProfile,
    ACTION_WEIGHTS,
)
from app.storage import ModelStorage


def build_small_dataset():
    behaviors = [
        UserBehavior("u1", "a", "like"),
        UserBehavior("u1", "b", "view"),
        UserBehavior("u1", "c", "collect"),
        UserBehavior("u2", "a", "view"),
        UserBehavior("u2", "c", "like"),
        UserBehavior("u2", "d", "view"),
        UserBehavior("u3", "b", "like"),
        UserBehavior("u3", "c", "view"),
        UserBehavior("u3", "d", "collect"),
        UserBehavior("u4", "a", "view"),
        UserBehavior("u4", "b", "view"),
        UserBehavior("u4", "d", "like"),
    ]
    items = [
        ItemProfile("a", ["tech", "ai"]),
        ItemProfile("b", ["tech", "programming"]),
        ItemProfile("c", ["sports", "football"]),
        ItemProfile("d", ["sports", "basketball"]),
        ItemProfile("e", ["tech", "ai"]),
    ]
    return behaviors, items


class TestItemCFSimilarity(unittest.TestCase):
    def setUp(self):
        behaviors, items = build_small_dataset()
        self.rec = ItemCFRecommender(similarity_threshold=0.0)
        self.rec.train(behaviors, items)

    def test_matrix_dimensions(self):
        n_users = len(self.rec.user_to_idx)
        n_items = len(self.rec.item_to_idx)
        self.assertEqual(self.rec.user_item_matrix.shape, (n_users, n_items))
        self.assertEqual(self.rec.item_similarity.shape, (n_items, n_items))

    def test_user_item_matrix_values(self):
        u1 = self.rec.user_to_idx["u1"]
        a = self.rec.item_to_idx["a"]
        b = self.rec.item_to_idx["b"]
        c = self.rec.item_to_idx["c"]
        d = self.rec.item_to_idx["d"]
        self.assertAlmostEqual(self.rec.user_item_matrix[u1, a], ACTION_WEIGHTS["like"])
        self.assertAlmostEqual(self.rec.user_item_matrix[u1, b], ACTION_WEIGHTS["view"])
        self.assertAlmostEqual(self.rec.user_item_matrix[u1, c], ACTION_WEIGHTS["collect"])
        self.assertAlmostEqual(self.rec.user_item_matrix[u1, d], 0.0)

    def test_similarity_symmetry(self):
        sim = self.rec.item_similarity
        n = sim.shape[0]
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(sim[i, j], sim[j, i], places=10)

    def test_similarity_diagonal_zero(self):
        sim = self.rec.item_similarity
        n = sim.shape[0]
        for i in range(n):
            self.assertAlmostEqual(sim[i, i], 0.0, places=10)

    def test_similarity_range(self):
        sim = self.rec.item_similarity
        self.assertTrue(np.all(sim >= -1.0 - 1e-9))
        self.assertTrue(np.all(sim <= 1.0 + 1e-9))

    def test_relative_similarity_order(self):
        a_idx = self.rec.item_to_idx["a"]
        b_idx = self.rec.item_to_idx["b"]
        c_idx = self.rec.item_to_idx["c"]
        d_idx = self.rec.item_to_idx["d"]
        sim = self.rec.item_similarity

        sim_ab = sim[a_idx, b_idx]
        sim_ac = sim[a_idx, c_idx]
        sim_bd = sim[b_idx, d_idx]
        sim_cd = sim[c_idx, d_idx]

        self.assertGreater(sim_ac, sim_ab)
        self.assertGreater(sim_bd, sim_cd)
        self.assertAlmostEqual(sim_ab, -0.8217814036133182, places=6)
        self.assertAlmostEqual(sim_ac, 0.6124713059090252, places=6)
        self.assertAlmostEqual(sim_bd, 0.7195945198695816, places=6)
        self.assertAlmostEqual(sim_cd, -0.6194178811709941, places=6)

    def test_adjusted_cosine_hand_calculated(self):
        a_idx = self.rec.item_to_idx["a"]
        c_idx = self.rec.item_to_idx["c"]
        sim = self.rec.item_similarity

        u_means = self.rec.user_item_matrix.mean(axis=1, keepdims=True)
        u_means_safe = np.where(u_means == 0, 1e-9, u_means)
        adj = self.rec.user_item_matrix - u_means_safe
        norms = np.linalg.norm(adj, axis=0, keepdims=True)
        norms_safe = np.where(norms == 0, 1e-9, norms)
        normed = adj / norms_safe
        expected_sim = (normed[:, a_idx] @ normed[:, c_idx])

        self.assertAlmostEqual(sim[a_idx, c_idx], expected_sim, places=10)


class TestItemCFRecommend(unittest.TestCase):
    def setUp(self):
        behaviors, items = build_small_dataset()
        self.rec = ItemCFRecommender(similarity_threshold=0.0)
        self.rec.train(behaviors, items)

    def test_recommend_returns_topn(self):
        results = self.rec.recommend("u1", top_n=2)
        self.assertLessEqual(len(results), 2)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_recommend_excludes_interacted(self):
        results = self.rec.recommend("u1", top_n=10)
        interacted = {"a", "b", "c"}
        for r in results:
            self.assertNotIn(r.item_id, interacted)

    def test_recommend_score_calculation_u1(self):
        results = self.rec.recommend("u1", top_n=5)
        d_result = next((r for r in results if r.item_id == "d"), None)
        self.assertIsNotNone(d_result)

        a_idx = self.rec.item_to_idx["a"]
        b_idx = self.rec.item_to_idx["b"]
        c_idx = self.rec.item_to_idx["c"]
        d_idx = self.rec.item_to_idx["d"]
        u1_idx = self.rec.user_to_idx["u1"]

        expected = 0.0
        for i_idx, w in [
            (a_idx, self.rec.user_item_matrix[u1_idx, a_idx]),
            (b_idx, self.rec.user_item_matrix[u1_idx, b_idx]),
            (c_idx, self.rec.user_item_matrix[u1_idx, c_idx]),
        ]:
            sim = self.rec.item_similarity[i_idx, d_idx]
            if sim > 0:
                expected += sim * w

        self.assertAlmostEqual(d_result.score, expected, places=10)

    def test_recommend_reason_contains_source_items(self):
        results = self.rec.recommend("u1", top_n=5)
        for r in results:
            if "因为你读过" in r.reason:
                self.assertTrue(r.reason.startswith("因为你读过"))

    def test_recommend_deterministic(self):
        results1 = self.rec.recommend("u2", top_n=5)
        results2 = self.rec.recommend("u2", top_n=5)
        self.assertEqual(len(results1), len(results2))
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1.item_id, r2.item_id)
            self.assertAlmostEqual(r1.score, r2.score, places=10)
            self.assertEqual(r1.reason, r2.reason)


class TestItemCFColdStart(unittest.TestCase):
    def setUp(self):
        behaviors, items = build_small_dataset()
        self.rec = ItemCFRecommender(similarity_threshold=0.0)
        self.rec.train(behaviors, items)

    def test_new_user_cold_start_popular(self):
        results = self.rec.recommend("new_user_xyz", top_n=3)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.reason, "热门推荐")
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_popular_order_correct(self):
        expected_pop = {
            "a": ACTION_WEIGHTS["like"] + ACTION_WEIGHTS["view"] + ACTION_WEIGHTS["view"],
            "b": ACTION_WEIGHTS["view"] + ACTION_WEIGHTS["like"] + ACTION_WEIGHTS["view"],
            "c": ACTION_WEIGHTS["collect"] + ACTION_WEIGHTS["like"] + ACTION_WEIGHTS["view"],
            "d": ACTION_WEIGHTS["view"] + ACTION_WEIGHTS["collect"] + ACTION_WEIGHTS["like"],
        }
        for iid, expected in expected_pop.items():
            self.assertAlmostEqual(self.rec.item_popularity[iid], expected)

    def test_tag_fallback_used_when_no_cf_scores(self):
        strict_rec = ItemCFRecommender(similarity_threshold=0.99)
        behaviors, items = build_small_dataset()
        strict_rec.train(behaviors, items)
        results = strict_rec.recommend("u1", top_n=5)
        self.assertGreaterEqual(len(results), 0)


class TestItemCFSimilarItems(unittest.TestCase):
    def setUp(self):
        behaviors, items = build_small_dataset()
        self.rec = ItemCFRecommender(similarity_threshold=0.0)
        self.rec.train(behaviors, items)

    def test_similar_items_excludes_self(self):
        results = self.rec.get_similar_items("a", top_n=10)
        for r in results:
            self.assertNotEqual(r.item_id, "a")

    def test_similar_items_sorted_descending(self):
        results = self.rec.get_similar_items("c", top_n=5)
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_similar_items_tag_fallback_for_new_item(self):
        results = self.rec.get_similar_items("e", top_n=5)
        self.assertGreater(len(results), 0)
        item_ids = [r.item_id for r in results]
        self.assertIn("a", item_ids)

    def test_similar_items_deterministic(self):
        r1 = self.rec.get_similar_items("b", top_n=5)
        r2 = self.rec.get_similar_items("b", top_n=5)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a.item_id, b.item_id)
            self.assertAlmostEqual(a.similarity, b.similarity, places=10)


class TestItemCFPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_model(self):
        behaviors, items = build_small_dataset()
        rec1 = ItemCFRecommender(similarity_threshold=0.0)
        rec1.train(behaviors, items)

        storage = ModelStorage(data_dir=self.tmpdir)
        storage.save_model(rec1)
        storage.save_behaviors(behaviors)
        storage.save_items(items)

        rec2 = storage.load_model()
        self.assertIsNotNone(rec2)
        self.assertTrue(rec2.is_trained)

        self.assertEqual(rec1.user_to_idx, rec2.user_to_idx)
        self.assertEqual(rec1.item_to_idx, rec2.item_to_idx)

        np.testing.assert_array_almost_equal(
            rec1.user_item_matrix, rec2.user_item_matrix, decimal=10
        )
        np.testing.assert_array_almost_equal(
            rec1.item_similarity, rec2.item_similarity, decimal=10
        )

        recs1 = rec1.recommend("u1", top_n=5)
        recs2 = rec2.recommend("u1", top_n=5)
        self.assertEqual(len(recs1), len(recs2))
        for a, b in zip(recs1, recs2):
            self.assertEqual(a.item_id, b.item_id)
            self.assertAlmostEqual(a.score, b.score, places=10)

    def test_load_nonexistent_returns_none(self):
        storage = ModelStorage(data_dir=self.tmpdir)
        self.assertIsNone(storage.load_model())


class TestItemCFMultipleRunsDeterministic(unittest.TestCase):
    def test_multiple_training_runs_produce_identical_results(self):
        behaviors, items = build_small_dataset()

        results_list = []
        for _ in range(5):
            rec = ItemCFRecommender(similarity_threshold=0.0)
            rec.train(behaviors, items)
            recs = rec.recommend("u3", top_n=5)
            sims = rec.get_similar_items("a", top_n=5)
            results_list.append((recs, sims, rec.item_similarity.copy()))

        for i in range(1, len(results_list)):
            prev_recs, prev_sims, prev_sim_matrix = results_list[i - 1]
            curr_recs, curr_sims, curr_sim_matrix = results_list[i]

            for a, b in zip(prev_recs, curr_recs):
                self.assertEqual(a.item_id, b.item_id)
                self.assertAlmostEqual(a.score, b.score, places=10)
                self.assertEqual(a.reason, b.reason)

            for a, b in zip(prev_sims, curr_sims):
                self.assertEqual(a.item_id, b.item_id)
                self.assertAlmostEqual(a.similarity, b.similarity, places=10)

            np.testing.assert_array_almost_equal(prev_sim_matrix, curr_sim_matrix, decimal=10)


if __name__ == "__main__":
    unittest.main()
