"""Validate workflow settings and expand each model's actual architecture grid.

This module deliberately has no single-cell or deep-learning imports, so building
the Snakemake DAG does not initialize a GPU or load a dataset.
"""

from copy import deepcopy
from itertools import product
import re


MODELS = ("scvi", "mrvi", "ldvae")
FEATURES = ("full", "hvg")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
RESERVED_MODEL_KWARGS = {
    "adata", "n_hidden", "n_latent", "n_layers", "n_latent_u",
    "encoder_n_hidden", "encoder_n_layers",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value, name, minimum=1):
    require(type(value) is int and value >= minimum,
            f"{name} must be an integer >= {minimum}.")


def unique_list(value, name):
    require(isinstance(value, list) and bool(value), f"{name} must be a nonempty YAML list.")
    require(len(value) == len(set(value)), f"{name} contains duplicate values.")


def validate_config(config):
    """Return a validated copy; paths are checked by Snakemake as rule inputs."""
    cfg = deepcopy(config)
    for key in ("selected_datasets", "models", "features", "seeds"):
        require(key in cfg, f"Missing configuration: {key}")
        unique_list(cfg[key], key)
    require(set(cfg["models"]) <= set(MODELS), f"models must be drawn from {MODELS}.")
    require(set(cfg["features"]) <= set(FEATURES), "features must contain full and/or hvg.")
    require(cfg.get("accelerator") in ("cpu", "gpu"), "accelerator must be cpu or gpu.")
    for seed in cfg["seeds"]:
        positive_int(seed, "seeds", minimum=0)
    for key in ("output_dir", "environment"):
        require(isinstance(cfg.get(key), str) and cfg[key].strip(), f"{key} must be a path.")
        require("{" not in cfg[key] and "}" not in cfg[key], f"{key} cannot contain braces.")
    require(cfg["output_dir"].rstrip("/") not in ("", ".", ".."),
            "output_dir must name a dedicated results directory.")
    cfg["output_dir"] = cfg["output_dir"].rstrip("/")

    grid = cfg.get("grid", {})
    grid_keys = ["n_hidden", "n_latent", "n_layers"]
    if "mrvi" in cfg["models"]:
        grid_keys.append("n_latent_u")
    for key in grid_keys:
        unique_list(grid.get(key), f"grid.{key}")
        for value in grid[key]:
            positive_int(value, f"grid.{key}")

    require(isinstance(cfg.get("datasets"), dict), "datasets must be a mapping.")
    for name in cfg["selected_datasets"]:
        require(isinstance(name, str) and SAFE_NAME.fullmatch(name),
                "Dataset IDs may contain letters, digits, underscores and hyphens.")
        require(name in cfg["datasets"], f"Dataset {name!r} is not configured.")
        ds = cfg["datasets"][name]
        for key in ("path", "counts_layer", "batch_key", "label_key"):
            require(isinstance(ds.get(key), str) and ds[key].strip(),
                    f"datasets.{name}.{key} must be a nonempty string.")
        if "mrvi" in cfg["models"]:
            require(isinstance(ds.get("sample_key"), str) and ds["sample_key"].strip(),
                    f"datasets.{name}.sample_key is required for MrVI.")
        if "sample_columns" in ds:
            unique_list(ds["sample_columns"], f"datasets.{name}.sample_columns")
            require(all(isinstance(key, str) and key for key in ds["sample_columns"]),
                    "sample_columns must contain metadata column names.")
            require(ds.get("sample_key") not in ds["sample_columns"],
                    "sample_key must differ from the columns used to derive it.")
        require(isinstance(ds.get("exclude_labels", []), list),
                f"datasets.{name}.exclude_labels must be a list.")
        pre = preprocessing_settings(cfg, name)
        positive_int(pre["min_cells_per_gene"], "preprocessing.min_cells_per_gene")
        positive_int(pre["min_cells_per_batch"], "preprocessing.min_cells_per_batch")
        positive_int(pre["n_top_genes"], "preprocessing.n_top_genes", minimum=2)
        require(0 <= pre["min_genes_quantile"] < pre["max_counts_quantile"] <= 1,
                "QC quantiles must satisfy 0 <= min_genes_quantile < max_counts_quantile <= 1.")
        mt = pre["max_mito_pct"]
        require(mt is None or (type(mt) in (int, float) and 0 <= mt <= 100),
                "max_mito_pct must be null or a percentage between 0 and 100.")

    training = cfg["training"]
    for key in ("max_epochs", "batch_size"):
        positive_int(training[key], f"training.{key}")
    require(type(training["early_stopping"]) is bool,
            "training.early_stopping must be a YAML boolean, not a quoted string.")
    require(0 < training["train_size"] < 1, "training.train_size must lie between 0 and 1.")
    require(training["mrvi_representation"] in ("u", "z"), "mrvi_representation must be u or z.")
    for model, kwargs in cfg.get("model_kwargs", {}).items():
        require(model in MODELS and isinstance(kwargs, dict), "Invalid model_kwargs mapping.")
        require(not (set(kwargs) & RESERVED_MODEL_KWARGS),
                "Architecture parameters belong in grid, not model_kwargs.")
    for key, value in cfg["resources"].items():
        positive_int(value, f"resources.{key}")
    for name in cfg["selected_datasets"]:
        evaluation = dict(cfg["evaluation"])
        evaluation.update(cfg["datasets"][name].get("evaluation", {}))
        validate_evaluation(evaluation)
    require(isinstance(cfg["plots"], list) and set(cfg["plots"]) <= {"umap", "tsne"},
            "plots must be a list containing umap and/or tsne, or [].")
    require(len(cfg["plots"]) == len(set(cfg["plots"])), "plots contains duplicates.")
    return cfg


def validate_evaluation(evaluation):
    positive_int(evaluation["n_neighbors"], "evaluation.n_neighbors", minimum=2)
    positive_int(evaluation["n_pcs"], "evaluation.n_pcs", minimum=2)
    positive_int(evaluation["lisi_k0"], "evaluation.lisi_k0", minimum=2)
    require(type(evaluation["strict_metrics"]) is bool, "strict_metrics must be a YAML boolean.")
    require(evaluation["pcr_input"] in ("counts", "log_normalized"),
            "evaluation.pcr_input must be counts or log_normalized.")
    require(evaluation["isolated_labels"] in ("auto", "off"), "isolated_labels must be auto or off.")
    require(evaluation["trajectory"] in ("auto", "off"), "trajectory must be auto or off.")
    maximum = evaluation["max_cells"]
    if maximum is not None:
        positive_int(maximum, "evaluation.max_cells", minimum=100)
    unique_list(evaluation["resolutions"], "evaluation.resolutions")
    require(all(type(value) in (int, float) and 0 < value < float("inf")
                for value in evaluation["resolutions"]), "Leiden resolutions must be finite and positive.")
    subsample = evaluation["lisi_subsample"]
    require(subsample is None or (type(subsample) is int and 1 <= subsample <= 100),
            "evaluation.lisi_subsample must be null or an integer percentage from 1 to 100.")


def preprocessing_settings(cfg, dataset):
    settings = deepcopy(cfg["preprocessing"])
    settings.update(cfg["datasets"][dataset].get("preprocessing", {}))
    return settings


def build_runs(cfg):
    """Expand z/u independently only for MrVI; never multiply duplicate axis lists."""
    runs = []
    grid = cfg["grid"]
    for dataset, feature, model in product(cfg["selected_datasets"], cfg["features"], cfg["models"]):
        u_values = grid["n_latent_u"] if model == "mrvi" else [None]
        for hidden, latent, layers, latent_u, seed in product(
            grid["n_hidden"], grid["n_latent"], grid["n_layers"], u_values, cfg["seeds"]
        ):
            run_id = f"h{hidden}_z{latent}_l{layers}"
            if latent_u is not None:
                run_id += f"_u{latent_u}"
            run_id += f"_s{seed}"
            runs.append({
                "dataset": dataset, "feature": feature, "model": model, "run_id": run_id,
                "n_hidden": hidden, "n_latent": latent, "n_layers": layers,
                "n_latent_u": latent_u, "seed": seed,
            })
    return runs


def run_settings(cfg, run):
    dataset = cfg["datasets"][run["dataset"]]
    return {
        "run": run, "dataset": {key: dataset[key] for key in ("batch_key", "sample_key")
                                  if key in dataset},
        "training": cfg["training"], "accelerator": cfg["accelerator"],
        "model_kwargs": cfg.get("model_kwargs", {}).get(run["model"], {}),
    }
