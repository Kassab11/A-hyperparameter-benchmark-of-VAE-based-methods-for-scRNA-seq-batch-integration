"""Write synthetic integer counts for config/smoke.yaml (no downloaded data)."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".test-data/smoke.h5ad")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    n_genes = 512
    # Three cell types in each of six samples; two batches, 360 cells in total.
    sample = np.repeat(np.arange(6), 60)
    batch = sample // 3
    label = np.tile(np.repeat(np.arange(3), 20), 6)
    baseline = rng.lognormal(mean=-0.8, sigma=1.0, size=n_genes)
    rates = np.broadcast_to(baseline, (len(sample), n_genes)).copy()
    for cell_type in range(3):
        rates[label == cell_type, cell_type * 40:(cell_type + 1) * 40] *= 4
    rates[batch == 1, 160:240] *= 1.8
    rates *= rng.lognormal(mean=0, sigma=0.15, size=(len(sample), 1))
    counts = sparse.csr_matrix(rng.poisson(rates).astype(np.float32))
    obs = pd.DataFrame({
        "batch": pd.Categorical([f"batch_{value}" for value in batch]),
        "sample": pd.Categorical([f"sample_{value}" for value in sample]),
        "cell_type": pd.Categorical([f"type_{value}" for value in label]),
    }, index=[f"cell_{i:04d}" for i in range(len(sample))])
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(n_genes)])
    data = ad.AnnData(X=counts, obs=obs, var=var)
    data.layers["counts"] = counts.copy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression="gzip")
    print(f"Wrote {data.n_obs} cells x {data.n_vars} genes to {output}")


if __name__ == "__main__":
    main()
