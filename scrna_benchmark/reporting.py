"""Plots and combined tables generated from explicit Snakemake inputs."""

from pathlib import Path
import json

from .io import read_json, write_json
from .metrics import ALL_METRICS


def plot_embedding(input_path, output_path, kind, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import anndata as ad
    import scanpy as sc

    data = ad.read_h5ad(input_path)
    training = json.loads(data.uns["benchmark_training"])
    seed, representation = training["seed"], training["representation"]
    if kind == "umap":
        sc.tl.umap(data, random_state=seed)
    elif kind == "tsne":
        sc.tl.tsne(data, use_rep=representation, random_state=seed,
                   perplexity=min(30, max(1, (data.n_obs - 1) / 3)))
    else:
        raise ValueError(f"Unknown plot kind: {kind}")
    ds = settings["dataset"]
    figure = sc.pl.embedding(data, basis=kind, color=[ds["label_key"], ds["batch_key"]],
                             show=False, return_fig=True, frameon=False)
    figure.suptitle(f"{training['dataset']} | {training['method']} | {training['feature']} | {training['run_id']}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate(inputs, csv_path, xlsx_path, best_path, status_path, config_path, settings):
    import pandas as pd

    rows, statuses = [], []
    identities = set()
    for path in inputs:
        result = read_json(path)
        training = result["training"]
        identity = tuple(training[key] for key in ("dataset", "feature", "model", "run_id"))
        if identity in identities:
            raise ValueError(f"Duplicate run in aggregation: {identity}")
        identities.add(identity)
        row = {
            "Dataset": training["dataset"], "Method": training["method"],
            "Features": training["feature"], "Run ID": training["run_id"],
            "n_hidden": training["n_hidden"], "n_latent": training["n_latent"],
            "n_layers": training["n_layers"], "n_latent_u": training["n_latent_u"],
            "Seed": training["seed"], "Training Time (s)": training["training_seconds"],
            "Epochs Completed": training["epochs_completed"],
            "GPU Peak Memory (GiB)": training["peak_gpu_memory_gib"],
            "Memory Measurement": training["memory_measurement"],
            "Cells": training["n_cells"], "Genes": training["n_genes"],
            "Evaluation Cells": result["n_evaluation_cells"],
            "Embedding Dimensions": training["embedding_dimensions"],
            "Clustering Resolution": result["clustering_resolution"],
            **result["composites"],
            **{name: result["metrics"][name]["value"] for name in ALL_METRICS},
            "Evaluation Complete": result["complete"], "Metrics File": str(path),
        }
        rows.append(row)
        for name in ALL_METRICS:
            statuses.append({"Dataset": row["Dataset"], "Method": row["Method"],
                             "Features": row["Features"], "Run ID": row["Run ID"],
                             "Metric": name, **result["metrics"][name]})
    if not rows:
        raise ValueError("There are no metric files to aggregate.")
    results = pd.DataFrame(rows).sort_values(
        ["Dataset", "Method", "Seed", "Overall"], ascending=[True, True, True, False], na_position="last")
    # Select within each seed; do not cherry-pick the best random seed.
    best = results.dropna(subset=["Overall"]).groupby(
        ["Dataset", "Method", "Seed"], sort=False, as_index=False).head(1)
    status = pd.DataFrame(statuses)
    for path in (csv_path, xlsx_path, best_path, status_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    best.to_csv(best_path, index=False)
    status.to_csv(status_path, index=False, sep="\t")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="All runs", index=False)
        best.to_excel(writer, sheet_name="Best per model and seed", index=False)
        status.to_excel(writer, sheet_name="Metric status", index=False)
    write_json(config_path, settings)
