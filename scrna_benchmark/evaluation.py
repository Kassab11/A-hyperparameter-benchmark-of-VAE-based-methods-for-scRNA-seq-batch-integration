"""Evaluate saved embeddings without loading a trained model or using a GPU."""

import json
import logging

import anndata as ad
import numpy as np
import scanpy as sc
import scib

from .io import names_hash, read_json, versions, write_h5ad, write_json
from .metrics import ALL_METRICS, composite_scores


LOGGER = logging.getLogger(__name__)


def evaluate(reference_path, embedding_path, training_path, output_path, evaluated_path, settings, threads):
    metadata = read_json(training_path)
    data = ad.read_h5ad(embedding_path)
    reference = ad.read_h5ad(reference_path)
    if (not reference.obs_names.equals(data.obs_names)
            or names_hash(data.obs_names) != metadata["cell_ids_sha256"]):
        raise ValueError("The embedding and full-gene reference have different cells or cell order.")
    ds, options = settings["dataset"], settings["evaluation"]
    batch, label = ds["batch_key"], ds["label_key"]
    representation, seed = metadata["representation"], metadata["seed"]
    maximum = options["max_cells"]
    if maximum is not None and data.n_obs > maximum:
        positions = np.sort(np.random.default_rng(seed).choice(data.n_obs, maximum, replace=False))
        data, reference = data[positions].copy(), reference[positions].copy()
    for key in (batch, label):
        data.obs[key] = data.obs[key].astype("category").cat.remove_unused_categories()
        reference.obs[key] = reference.obs[key].astype("category").cat.remove_unused_categories()
    if data.obs[batch].nunique() < 2 or data.obs[label].nunique() < 2:
        raise ValueError("Evaluation needs at least two batches and labels, including after subsampling.")
    reference.X = reference.X.astype(np.float32)
    if options["pcr_input"] == "log_normalized":
        sc.pp.normalize_total(reference, target_sum=1e4)
        sc.pp.log1p(reference)
    np.random.seed(seed)
    sc.pp.neighbors(data, use_rep=representation, n_neighbors=min(options["n_neighbors"], data.n_obs - 1),
                    random_state=seed)
    metrics = {}

    def skip(name, reason, status="not_applicable"):
        metrics[name] = {"value": None, "status": status, "reason": reason}
        LOGGER.info("%s: %s (%s)", name, status, reason)

    def score(name, callback):
        try:
            value = float(callback())
            if not np.isfinite(value):
                raise ValueError("Metric returned a non-finite score.")
            metrics[name] = {"value": value, "status": "ok", "reason": ""}
            LOGGER.info("%s: %.6f", name, value)
        except Exception as error:
            metrics[name] = {"value": None, "status": "error", "reason": f"{type(error).__name__}: {error}"}
            LOGGER.exception("Failed metric %s", name)

    batches_per_label = data.obs.groupby(label, observed=True)[batch].nunique()
    if (batches_per_label > 1).any():
        score("Batch ASW", lambda: scib.me.silhouette_batch(
            data, batch_key=batch, label_key=label, embed=representation, scale=True))
    else:
        skip("Batch ASW", "No cell-type label is shared between batches.")
    score("PCR Batch", lambda: scib.me.pcr_comparison(
        reference, data, covariate=batch, embed=representation,
        n_comps=min(options["n_pcs"], reference.n_obs - 1, reference.n_vars - 1), n_threads=threads))
    score("Graph Connectivity", lambda: scib.me.graph_connectivity(data, label_key=label))
    lisi_options = dict(type_="knn", use_rep=representation, n_cores=threads,
                        k0=min(options["lisi_k0"], data.n_obs - 1),
                        subsample=options["lisi_subsample"], scale=True)
    score("iLISI", lambda: scib.me.ilisi_graph(data, batch_key=batch, **lisi_options))
    score("cLISI", lambda: scib.me.clisi_graph(data, label_key=label, **lisi_options))

    # Select the resolution by NMI as in the benchmark. Starting at -inf also
    # handles the valid edge case where every candidate has NMI == 0.
    resolution = None
    try:
        best_score, best_clusters = -np.inf, None
        for candidate in options["resolutions"]:
            sc.tl.leiden(data, resolution=candidate, key_added="_benchmark_candidate",
                         random_state=seed, flavor="leidenalg")
            current = scib.me.nmi(data, cluster_key="_benchmark_candidate", label_key=label)
            if np.isfinite(current) and current > best_score:
                best_score, resolution = current, candidate
                best_clusters = data.obs["_benchmark_candidate"].copy()
        if best_clusters is None:
            raise ValueError("No clustering resolution produced a finite NMI.")
        data.obs["benchmark_leiden"] = best_clusters
        del data.obs["_benchmark_candidate"]
        score("NMI", lambda: scib.me.nmi(data, cluster_key="benchmark_leiden", label_key=label))
        score("ARI", lambda: scib.me.ari(data, cluster_key="benchmark_leiden", label_key=label))
    except Exception as error:
        for name in ("NMI", "ARI"):
            metrics[name] = {"value": None, "status": "error", "reason": f"Clustering: {error}"}
        LOGGER.exception("Clustering failed")
    score("Label ASW", lambda: scib.me.silhouette(data, label_key=label, embed=representation, scale=True))

    if options["isolated_labels"] == "off":
        for name in ("Isolated Label F1", "Isolated Label ASW"):
            skip(name, "Disabled in evaluation configuration.", "disabled")
    elif batches_per_label.min() == data.obs[batch].nunique():
        for name in ("Isolated Label F1", "Isolated Label ASW"):
            skip(name, "All labels occur in every batch; there are no isolated labels.")
    else:
        score("Isolated Label F1", lambda: scib.me.isolated_labels_f1(
            data, batch_key=batch, label_key=label, embed=None,
            resolutions=options["resolutions"], verbose=False, random_state=seed))
        score("Isolated Label ASW", lambda: scib.me.isolated_labels_asw(
            data, batch_key=batch, label_key=label, embed=representation, verbose=False))
    pseudotime = ds.get("pseudotime_key", "dpt_pseudotime")
    if options["trajectory"] == "off":
        skip("Trajectory Conservation", "Disabled in evaluation configuration.", "disabled")
    elif (not pseudotime or pseudotime not in reference.obs
          or reference.obs[pseudotime].dropna().nunique() < 2):
        skip("Trajectory Conservation", "No varying reference pseudotime annotation is available.")
    else:
        score("Trajectory Conservation", lambda: scib.me.trajectory_conservation(
            reference, data, label_key=label, pseudotime_key=pseudotime))

    failures = [name for name in ALL_METRICS if metrics[name]["status"] == "error"]
    result = {
        "training": metadata, "evaluation": options, "evaluation_versions": versions(),
        "n_evaluation_cells": data.n_obs, "evaluation_cell_ids_sha256": names_hash(data.obs_names),
        "clustering_resolution": resolution, "metrics": metrics,
        "composites": composite_scores(metrics), "complete": not failures,
    }
    if failures and options["strict_metrics"]:
        raise RuntimeError("Evaluation failed for: " + ", ".join(failures)
                           + ". See this rule's log. No successful result is recorded.")
    data.uns["benchmark_evaluation"] = json.dumps(result, sort_keys=True)
    write_h5ad(evaluated_path, data)
    write_json(output_path, result)
