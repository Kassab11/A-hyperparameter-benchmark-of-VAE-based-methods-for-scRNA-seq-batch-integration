"""Model-specific scvi-tools 1.3 adapters; one process per training run."""

from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import os
import time

from .io import names_hash, versions, write_h5ad, write_json


LOGGER = logging.getLogger(__name__)
REPRESENTATIONS = {"scvi": "X_scVI", "mrvi": "X_mrVI", "ldvae": "X_ldvae"}


def train_model(input_path, model_dir, embedding_path, metadata_path, settings):
    # These variables must be set before scvi imports/initializes JAX.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if settings["accelerator"] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    import anndata as ad
    import numpy as np
    import scvi

    if scvi.__version__ != "1.3.0":
        raise RuntimeError("This workflow targets scvi-tools 1.3.0, including its JAX MrVI API. "
                           "Use the supplied environment.yml or envs/gpu.yaml.")
    run, ds, training = settings["run"], settings["dataset"], settings["training"]
    scvi.settings.seed = run["seed"]
    data = ad.read_h5ad(input_path)
    preprocessing = json.loads(data.uns["benchmark_preprocessing"])
    if preprocessing["dataset"] != run["dataset"] or preprocessing["feature"] != run["feature"]:
        raise ValueError("Prepared dataset/feature metadata does not match the requested run.")
    model_name = run["model"]
    gpu = settings["accelerator"] == "gpu"
    torch = None
    jax_device = None
    kwargs = dict(settings["model_kwargs"])
    if model_name == "mrvi":
        import jax
        from scvi.external import MRVI

        devices = [device for device in jax.devices() if device.platform == ("gpu" if gpu else "cpu")]
        if not devices:
            raise RuntimeError("The requested device is unavailable to JAX. Use envs/gpu.yaml "
                               "for CUDA MrVI, or accelerator: cpu.")
        jax_device = devices[0]
        MRVI.setup_anndata(data, layer=None, batch_key=ds["batch_key"], sample_key=ds["sample_key"])
        model = MRVI(data, encoder_n_hidden=run["n_hidden"], n_latent=run["n_latent"],
                     encoder_n_layers=run["n_layers"], n_latent_u=run["n_latent_u"], **kwargs)
        backend = "jax"
    else:
        import torch
        from scvi.model import LinearSCVI, SCVI

        if gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch; use a GPU environment or accelerator: cpu.")
        cls = SCVI if model_name == "scvi" else LinearSCVI
        cls.setup_anndata(data, layer=None, batch_key=ds["batch_key"])
        model = cls(data, n_hidden=run["n_hidden"], n_latent=run["n_latent"],
                    n_layers=run["n_layers"], **kwargs)
        backend = "torch"
        if gpu:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    model.train(
        max_epochs=training["max_epochs"], batch_size=training["batch_size"],
        train_size=training["train_size"], early_stopping=training["early_stopping"],
        accelerator=settings["accelerator"], devices=1,
        enable_progress_bar=False,
    )
    if backend == "jax":
        # JAX dispatch is asynchronous; include completed training in the timing.
        import jax
        jax.block_until_ready(model.module.train_state.params)
    elif gpu:
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    peak_memory = None
    memory_measurement = "unavailable"
    if gpu and backend == "torch":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        memory_measurement = "torch.max_memory_allocated"
    elif gpu and backend == "jax":
        try:
            stats = jax_device.memory_stats() or {}
            if "peak_bytes_in_use" in stats:
                peak_memory = stats["peak_bytes_in_use"] / 1024**3
                memory_measurement = "jax.device.peak_bytes_in_use"
        except (RuntimeError, NotImplementedError):
            LOGGER.warning("JAX peak-memory statistics are unavailable on this device.")

    representation = REPRESENTATIONS[model_name]
    if model_name == "mrvi":
        latent = model.get_latent_representation(give_z=training["mrvi_representation"] == "z")
    else:
        latent = model.get_latent_representation()
    latent = np.asarray(latent)
    if latent.shape[0] != data.n_obs or not np.isfinite(latent).all():
        raise ValueError("Training produced an invalid embedding.")
    Path(model_dir).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    if model_name == "ldvae":
        model.get_loadings().to_csv(Path(model_dir) / "gene_loadings.csv")
    for name, history in (model.history or {}).items():
        history.to_csv(Path(model_dir) / f"history_{name}.csv")
    metadata = {
        **run, "method": {"scvi": "scVI", "mrvi": "MrVI", "ldvae": "LDVAE"}[model_name],
        "started_at": started_at, "training_seconds": training_seconds,
        "epochs_completed": int(model.trainer.current_epoch),
        "peak_gpu_memory_gib": peak_memory, "memory_measurement": memory_measurement,
        "backend": backend, "accelerator": settings["accelerator"],
        "representation": representation, "embedding_dimensions": int(latent.shape[1]),
        "prepared_path": str(Path(input_path).resolve()), "model_path": str(Path(model_dir).resolve()),
        "n_cells": data.n_obs, "n_genes": data.n_vars,
        "cell_ids_sha256": names_hash(data.obs_names), "gene_ids_sha256": names_hash(data.var_names),
        "settings": settings, "preprocessing": preprocessing, "versions": versions(),
    }
    # Avoid copying a full cell-by-gene expression matrix into every model folder.
    embedding = ad.AnnData(obs=data.obs.copy())
    embedding.obsm[representation] = latent
    embedding.uns["benchmark_training"] = json.dumps(metadata, sort_keys=True)
    write_h5ad(embedding_path, embedding)
    write_json(metadata_path, metadata)
    LOGGER.info("Finished %s/%s/%s/%s in %.2f seconds", run["dataset"], run["feature"],
                model_name, run["run_id"], training_seconds)
