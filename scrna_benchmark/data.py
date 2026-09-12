"""Prepare raw counts once per dataset, then derive HVGs from the same cells."""

from pathlib import Path
import json
import logging

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from .io import names_hash, write_h5ad, write_json


LOGGER = logging.getLogger(__name__)


def validate_counts(matrix):
    if matrix is None:
        raise ValueError("The configured counts source is empty. Select a counts layer, X, or raw.")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).reshape(-1)
    # Check every entry without materializing a dense copy of a sparse matrix.
    for start in range(0, values.size, 1_000_000):
        chunk = values[start:start + 1_000_000]
        if not np.isfinite(chunk).all() or (chunk < 0).any():
            raise ValueError("Counts must be finite and nonnegative.")
        if not np.allclose(chunk, np.rint(chunk), rtol=0, atol=1e-6):
            raise ValueError(
                "The selected expression matrix is not raw integer counts. "
                "Set counts_layer to the correct layer or raw; normalized values are never rounded."
            )


def prepare_counts(input_path, output_path, metadata_path, settings):
    ds, pre = settings["dataset"], settings["preprocessing"]
    source = ad.read_h5ad(input_path)
    counts_layer = ds["counts_layer"]
    if counts_layer == "raw":
        if source.raw is None:
            raise ValueError(f"{input_path} has no .raw expression matrix.")
        matrix, var = source.raw.X, source.raw.var.copy()
    elif counts_layer == "X":
        matrix, var = source.X, source.var.copy()
    else:
        if counts_layer not in source.layers:
            raise ValueError(f"Missing counts layer {counts_layer!r} in {input_path}.")
        matrix, var = source.layers[counts_layer], source.var.copy()
    validate_counts(matrix)
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    # Do not carry old graphs, embeddings, normalization, or scvi registries forward.
    data = ad.AnnData(X=matrix, obs=source.obs.copy(), var=var)
    del source
    if not data.obs_names.is_unique:
        raise ValueError("Cell IDs must be unique so embeddings can be aligned with their reference.")
    data.var_names_make_unique()
    original_shape = list(data.shape)
    if settings["require_sample"] and ds.get("sample_columns"):
        columns = ds["sample_columns"]
        missing = [key for key in columns if key not in data.obs]
        if missing:
            raise ValueError(f"Cannot derive sample IDs: metadata columns {missing} are missing.")
        valid = data.obs[columns].notna().all(axis=1)
        # A JSON tuple avoids collisions between mouse/tissue names containing separators.
        sample_ids = data.obs.loc[valid, columns].apply(
            lambda row: json.dumps([str(value) for value in row], separators=(",", ":")), axis=1)
        data.obs[ds["sample_key"]] = sample_ids.reindex(data.obs_names)
    required = [ds["batch_key"], ds["label_key"]]
    if settings["require_sample"]:
        required.append(ds["sample_key"])
    for key in required:
        if key not in data.obs:
            raise ValueError(f"Required metadata column {key!r} is missing from {input_path}.")
    keep = data.obs[required].notna().all(axis=1)
    keep &= ~data.obs[ds["label_key"]].isin(ds.get("exclude_labels", []))
    data = data[keep.to_numpy()].copy()
    if data.n_obs < 3:
        raise ValueError("Fewer than three cells remain after metadata/label filtering.")
    symbols_key = ds.get("gene_symbol_key")
    if symbols_key and symbols_key not in data.var:
        raise ValueError(f"Gene symbol column {symbols_key!r} is missing from .var.")
    symbols = data.var[symbols_key].fillna("").astype(str) if symbols_key else data.var_names
    data.var["mt"] = np.asarray(symbols.str.upper().str.startswith("MT-"))
    sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], percent_top=None, inplace=True, log1p=False)
    keep = np.zeros(data.n_obs, dtype=bool)
    thresholds = []
    for batch, positions in data.obs.groupby(ds["batch_key"], observed=True, sort=False).indices.items():
        group = data.obs.iloc[positions]
        if len(group) < pre["min_cells_per_batch"]:
            LOGGER.warning("Skipping batch %s: fewer than %s cells", batch, pre["min_cells_per_batch"])
            continue
        minimum = float(group["n_genes_by_counts"].quantile(pre["min_genes_quantile"]))
        maximum = float(group["total_counts"].quantile(pre["max_counts_quantile"]))
        accepted = ((group["n_genes_by_counts"] >= minimum)
                    & (group["total_counts"] <= maximum)
                    & (group["total_counts"] > 0))
        if pre["max_mito_pct"] is not None:
            accepted &= group["pct_counts_mt"] <= pre["max_mito_pct"]
        keep[positions] = accepted.to_numpy()
        thresholds.append({"batch": str(batch), "min_genes": minimum, "max_counts": maximum,
                           "cells_before": len(group), "cells_after": int(accepted.sum())})
    data = data[keep].copy()
    if data.n_obs < 3:
        raise ValueError("QC removed too many cells; inspect the configured per-batch thresholds.")
    sc.pp.filter_genes(data, min_cells=pre["min_cells_per_gene"])
    if data.n_vars < 2:
        raise ValueError("Fewer than two genes remain after gene filtering.")
    nonzero = np.asarray(data.X.sum(axis=1)).ravel() > 0
    data = data[nonzero].copy()
    if data.n_obs < 3:
        raise ValueError("Fewer than three cells retain counts after gene filtering.")
    for key in dict.fromkeys(required):
        data.obs[key] = data.obs[key].astype("category").cat.remove_unused_categories()
    if data.obs[ds["batch_key"]].nunique() < 2:
        raise ValueError("Batch integration requires at least two batches after QC.")
    if data.obs[ds["label_key"]].nunique() < 2:
        raise ValueError("Biological-conservation metrics require at least two cell-type labels.")
    if settings["require_sample"] and data.obs[ds["sample_key"]].nunique() < 2:
        raise ValueError("MrVI requires at least two samples after QC; check sample_key.")
    stat = Path(input_path).stat()
    metadata = {
        "dataset": settings["dataset_id"], "input_path": str(Path(input_path).resolve()),
        "input_bytes": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
        "counts_layer": counts_layer, "input_shape": original_shape, "feature": "full",
        "n_cells": data.n_obs, "n_genes": data.n_vars,
        "cell_ids_sha256": names_hash(data.obs_names), "gene_ids_sha256": names_hash(data.var_names),
        "settings": settings, "batch_qc": thresholds,
    }
    data.uns["benchmark_preprocessing"] = json.dumps(metadata, sort_keys=True)
    write_h5ad(output_path, data)
    write_json(metadata_path, metadata)
    LOGGER.info("Prepared %s: %s cells x %s count features", settings["dataset_id"], *data.shape)


def select_hvg(input_path, output_path, metadata_path, settings):
    data = ad.read_h5ad(input_path)
    n_top = min(settings["n_top_genes"], data.n_vars)
    if n_top < data.n_vars:
        sc.pp.highly_variable_genes(
            data, flavor="seurat_v3", n_top_genes=n_top,
            batch_key=settings["batch_key"], subset=True,
        )
    else:
        data.var["highly_variable"] = True
        LOGGER.warning("HVG limit covers all %s available genes; both feature sets contain all genes.", n_top)
    if (np.asarray(data.X.sum(axis=1)).ravel() == 0).any():
        raise ValueError("Some cells have zero counts across the HVGs. Increase n_top_genes; "
                         "cells are not dropped separately from the full-feature experiment.")
    metadata = json.loads(data.uns["benchmark_preprocessing"])
    metadata.update({"feature": "hvg", "n_genes": data.n_vars,
                     "gene_ids_sha256": names_hash(data.var_names), "hvg_settings": settings})
    data.uns["benchmark_preprocessing"] = json.dumps(metadata, sort_keys=True)
    write_h5ad(output_path, data)
    write_json(metadata_path, metadata)
