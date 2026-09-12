"""Input regressions for normalized matrices and processed AnnData raw counts."""

from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scrna_benchmark.data import prepare_counts, validate_counts


class CountTests(unittest.TestCase):
    def test_normalized_counts_are_rejected_in_dense_and_sparse_inputs(self):
        values = np.array([[0, 1.5], [2, 3]], dtype=np.float32)
        for matrix in (values, sparse.csr_matrix(values)):
            with self.subTest(matrix=type(matrix).__name__):
                with self.assertRaisesRegex(ValueError, "raw integer counts"):
                    validate_counts(matrix)

    def test_negative_and_nonfinite_counts_are_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    validate_counts(sparse.csr_matrix([[1, value]]))

    def test_raw_restores_all_genes_and_derives_samples_from_metadata(self):
        counts = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
        obs = pd.DataFrame({
            "batch": ["a"] * 4 + ["b"] * 4,
            "label": ["T", "B"] * 4,
            "mouse.id": ["mouse1"] * 4 + ["mouse2"] * 4,
            "tissue": ["blood", "blood", "lung", "lung"] * 2,
        }, index=[f"cell{i}" for i in range(8)])
        full = ad.AnnData(sparse.csr_matrix(counts), obs=obs,
                          var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]))
        full.raw = full.copy()
        processed = full[:, :2].copy()
        processed.X = sparse.csr_matrix(np.log1p(counts[:, :2]))
        settings = {
            "dataset_id": "test", "require_sample": True,
            "dataset": {"counts_layer": "raw", "batch_key": "batch", "label_key": "label",
                        "sample_key": "sample", "sample_columns": ["mouse.id", "tissue"]},
            "preprocessing": {"min_genes_quantile": 0, "max_counts_quantile": 1,
                              "min_cells_per_gene": 1, "min_cells_per_batch": 1,
                              "max_mito_pct": None},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed.write_h5ad(root / "input.h5ad")
            prepare_counts(root / "input.h5ad", root / "output.h5ad", root / "output.json", settings)
            prepared = ad.read_h5ad(root / "output.h5ad")
        self.assertEqual(prepared.shape, (8, 4))
        self.assertTrue(prepared.obs_names.equals(full.obs_names))
        np.testing.assert_array_equal(prepared.X.toarray(), counts)
        self.assertEqual(prepared.obs["sample"].nunique(), 4)
        self.assertNotIn("X_pca", prepared.obsm)


if __name__ == "__main__":
    unittest.main()
