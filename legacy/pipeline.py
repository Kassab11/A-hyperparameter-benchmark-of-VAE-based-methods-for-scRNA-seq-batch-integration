"""Historical single-run entry point; use the root Snakefile for new runs."""

import scanpy as sc
import numpy as np
import torch
import time
import warnings
import pandas as pd
import scib
import argparse

warnings.filterwarnings("ignore")

def preprocess_data(path, x_layer, batch_key, cell_type_key, use_hvg=False):
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    adata.X = adata.layers[x_layer]

    # Add QC annotations
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # Per-batch filtering
    filtered_batches = []
    batches = adata.obs[batch_key].unique()
    for batch in batches:
        batch_data = adata[adata.obs[batch_key] == batch].copy()
        min_genes = batch_data.obs["n_genes_by_counts"].quantile(0.01)
        max_counts = batch_data.obs["total_counts"].quantile(0.99)
        batch_data = batch_data[
            (batch_data.obs["n_genes_by_counts"] >= min_genes) &
            (batch_data.obs["total_counts"] <= max_counts)
        ]
        filtered_batches.append(batch_data)

    adata = sc.concat(filtered_batches, join="inner")
    sc.pp.filter_genes(adata, min_cells=3)

    if use_hvg:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=5000,
            subset=True,
            layer=x_layer,
            flavor="seurat_v3",
            batch_key=batch_key,
        )

    adata.layers["counts"] = np.rint(adata.layers[x_layer]).astype(np.int64)
    return adata.copy()


def run_model(model_type, adata, batch_key, n_hidden, n_latent, n_layers, n_latent_u=None):
    from scvi.model import SCVI
    from mrvi.model import MRVI
    from scvi.model import LDVAE

    model_cls = {"scvi": SCVI, "mrvi": MRVI, "ldvae": LDVAE}[model_type]

    model_cls.setup_anndata(adata, layer="counts", batch_key=batch_key)

    kwargs = {"adata": adata, "n_hidden": n_hidden, "n_latent": n_latent, "n_layers": n_layers}
    if model_type == "mrvi":
        kwargs["n_latent_u"] = n_latent_u

    model = model_cls(**kwargs)

    start = time.time()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()

    model.train(max_epochs=200)

    end = time.time()
    time_taken = end - start
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"Training Time: {time_taken:.2f}s | GPU Memory: {max_mem:.2f} GB")
    return model, time_taken, max_mem


def evaluate_model(model, adata_raw, adata, batch_key, cell_type_key, rep_key="X_scVI"):
    latent = model.get_latent_representation()
    adata.obsm[rep_key] = latent
    adata.layers["scvi_normalized"] = model.get_normalized_expression(library_size=1e4)

    results = {}

    try:
        results["Batch ASW"] = scib.me.silhouette_batch(adata, batch_key, cell_type_key, rep_key, scale=True)
    except Exception as e: results["Batch ASW"] = str(e)

    try:
        sc.pp.neighbors(adata, use_rep=rep_key)
        results["Graph Conn."] = scib.me.graph_connectivity(adata, label_key=cell_type_key)
    except Exception as e: results["Graph Conn."] = str(e)

    try:
        results["PCR Batch"] = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key, embed=rep_key)
    except Exception as e: results["PCR Batch"] = str(e)

    try:
        results["iLISI"] = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=rep_key)
    except Exception as e: results["iLISI"] = str(e)

    try:
        scib.me.cluster_optimal_resolution(adata, cluster_key="leiden", label_key=cell_type_key)
    except Exception: pass

    try:
        results["NMI"] = scib.me.nmi(adata, cluster_key="leiden", label_key=cell_type_key)
    except Exception as e: results["NMI"] = str(e)

    try:
        results["ARI"] = scib.me.ari(adata, cluster_key="leiden", label_key=cell_type_key)
    except Exception as e: results["ARI"] = str(e)

    try:
        results["Label ASW"] = scib.me.silhouette(adata, label_key=cell_type_key, embed=rep_key)
    except Exception as e: results["Label ASW"] = str(e)

    try:
        results["Isolated F1"] = scib.me.isolated_labels_f1(adata, batch_key=batch_key, embed=None, label_key=cell_type_key)
    except Exception as e: results["Isolated F1"] = str(e)

    try:
        results["Isolated ASW"] = scib.me.isolated_labels_asw(adata, batch_key=batch_key, label_key=cell_type_key, embed=rep_key)
    except Exception as e: results["Isolated ASW"] = str(e)

    try:
        results["cLISI"] = scib.me.clisi_graph(adata, label_key=cell_type_key, type_="embed", use_rep=rep_key)
    except Exception as e: results["cLISI"] = str(e)

    try:
        results["Trajectory"] = scib.me.trajectory_conservation(adata_raw, adata, label_key=cell_type_key)
    except Exception as e: results["Trajectory"] = str(e)

    df = pd.DataFrame.from_dict(results, orient="index", columns=["Score"])
    print(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scvi", "mrvi", "ldvae"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--x_layer", default="counts")
    parser.add_argument("--batch_key", required=True)
    parser.add_argument("--cell_type_key", required=True)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_latent", type=int, default=10)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_latent_u", type=int, default=10)
    parser.add_argument("--use_hvg", action="store_true")
    args = parser.parse_args()

    print(f"Loading and preprocessing {args.dataset}")
    adata = preprocess_data(args.dataset, args.x_layer, args.batch_key, args.cell_type_key, args.use_hvg)
    adata_raw = adata.copy()

    print(f"Training {args.model.upper()} model")
    model, _, _ = run_model(
        args.model, adata, args.batch_key, args.n_hidden, args.n_latent, args.n_layers, args.n_latent_u
    )

    print("Evaluating...")
    evaluate_model(model, adata_raw, adata, args.batch_key, args.cell_type_key)
