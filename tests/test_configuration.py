"""Configuration regressions; run manually on the execution machine."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from scrna_benchmark.config import build_runs, run_settings, validate_config


ROOT = Path(__file__).resolve().parents[1]


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.base = yaml.safe_load((ROOT / "config/config.yaml").read_text())

    def test_publication_grid_has_960_distinct_runs(self):
        override = yaml.safe_load((ROOT / "config/benchmark.yaml").read_text())
        runs = build_runs(validate_config(merge(self.base, override)))
        self.assertEqual(len(runs), 960)
        self.assertEqual(Counter(run["model"] for run in runs),
                         {"scvi": 240, "mrvi": 480, "ldvae": 240})
        identities = {(r["dataset"], r["feature"], r["model"], r["run_id"]) for r in runs}
        self.assertEqual(len(identities), 960)

    def test_mrvi_u_axis_does_not_duplicate_other_models(self):
        self.base["models"] = ["scvi", "ldvae"]
        del self.base["grid"]["n_latent_u"]
        runs = build_runs(validate_config(self.base))
        self.assertEqual(len(runs), 4)
        self.assertTrue(all(run["n_latent_u"] is None for run in runs))

    def test_multiple_seeds_get_distinct_outputs(self):
        self.base["seeds"] = [0, 42]
        runs = build_runs(validate_config(self.base))
        self.assertEqual(len(runs), 12)
        self.assertEqual({r["seed"] for r in runs}, {0, 42})
        self.assertEqual(len({(r["model"], r["feature"], r["run_id"]) for r in runs}), 12)

    def test_quoted_false_is_rejected(self):
        self.base["training"]["early_stopping"] = "False"
        with self.assertRaisesRegex(ValueError, "YAML boolean"):
            validate_config(self.base)

    def test_duplicate_axes_are_rejected(self):
        self.base["grid"]["n_hidden"] = [128, 128]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_config(self.base)

    def test_mrvi_requires_sample_metadata_configuration(self):
        del self.base["datasets"]["human_immune"]["sample_key"]
        with self.assertRaisesRegex(ValueError, "sample_key"):
            validate_config(self.base)

    def test_dataset_metric_overrides_are_validated(self):
        self.base["datasets"]["human_immune"]["evaluation"] = {"lisi_subsample": 0.5}
        with self.assertRaisesRegex(ValueError, "integer percentage"):
            validate_config(self.base)

    def test_smoke_uses_all_three_model_adapters_and_both_features(self):
        override = yaml.safe_load((ROOT / "config/smoke.yaml").read_text())
        runs = build_runs(validate_config(merge(self.base, override)))
        self.assertEqual(len(runs), 6)
        self.assertEqual({(r["model"], r["feature"]) for r in runs},
                         {(m, f) for m in ("scvi", "mrvi", "ldvae") for f in ("full", "hvg")})

    def test_metric_option_changes_do_not_change_training_parameters(self):
        run = build_runs(validate_config(self.base))[0]
        before = run_settings(self.base, run)
        self.base["datasets"]["human_immune"]["evaluation"] = {"trajectory": "off"}
        self.assertEqual(before, run_settings(self.base, run))


if __name__ == "__main__":
    unittest.main()
