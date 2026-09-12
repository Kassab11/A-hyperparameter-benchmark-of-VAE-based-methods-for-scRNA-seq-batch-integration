"""Protect group weighting and the distinction between unavailable and failed metrics."""

import unittest

from scrna_benchmark.metrics import ALL_METRICS, BATCH_METRICS, BIO_METRICS, composite_scores


def successful_metrics():
    return {name: {"value": 0.8 if name in BATCH_METRICS else 0.4, "status": "ok"}
            for name in ALL_METRICS}


class ScoreTests(unittest.TestCase):
    def test_groups_receive_equal_weight_despite_different_sizes(self):
        scores = composite_scores(successful_metrics())
        self.assertAlmostEqual(scores["Overall"], 0.6)
        self.assertEqual(scores["Batch Metric Count"], 4)
        self.assertEqual(scores["Bio Metric Count"], 7)

    def test_unavailable_trajectory_is_excluded_explicitly(self):
        metrics = successful_metrics()
        metrics["Trajectory Conservation"] = {"value": None, "status": "not_applicable"}
        scores = composite_scores(metrics)
        self.assertAlmostEqual(scores["Overall Bio"], 0.4)
        self.assertAlmostEqual(scores["Overall"], 0.6)
        self.assertEqual(scores["Bio Metric Count"], 6)

    def test_failed_metric_invalidates_its_group_and_overall(self):
        metrics = successful_metrics()
        metrics["iLISI"] = {"value": None, "status": "error"}
        scores = composite_scores(metrics)
        self.assertIsNone(scores["Overall"])
        self.assertIsNone(scores["Overall Batch"])
        self.assertAlmostEqual(scores["Overall Bio"], 0.4)

    def test_no_biological_metrics_does_not_become_zero(self):
        metrics = successful_metrics()
        for name in BIO_METRICS:
            metrics[name] = {"value": None, "status": "disabled"}
        scores = composite_scores(metrics)
        self.assertIsNone(scores["Overall Bio"])
        self.assertIsNone(scores["Overall"])
        self.assertEqual(scores["Bio Metric Count"], 0)

    def test_nonfinite_success_is_rejected(self):
        metrics = successful_metrics()
        metrics["iLISI"]["value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "not finite"):
            composite_scores(metrics)


if __name__ == "__main__":
    unittest.main()
