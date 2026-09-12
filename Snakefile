"""Shared preprocessing, model-specific training, scIB evaluation and reporting."""

from pathlib import Path
import json
import re
import sys

from snakemake.utils import min_version

min_version("9.0")
configfile: "config/config.yaml"

ROOT = Path(workflow.basedir).resolve()
sys.path.insert(0, str(ROOT))
from scrna_benchmark.config import build_runs, preprocessing_settings, run_settings, validate_config

CFG = validate_config(config)
OUT = CFG["output_dir"]
ENVIRONMENT = str(ROOT / CFG["environment"])
RUNNER = str(ROOT / "pipeline.py")
RUNS = build_runs(CFG)
LOOKUP = {(r["dataset"], r["feature"], r["model"], r["run_id"]): r for r in RUNS}
RUN_DIR = OUT + "/runs/{dataset}/{feature}/{model}/{run_id}"
PREPARED = OUT + "/prepared/{dataset}"
METRICS = [RUN_DIR.format(**run) + "/metrics.json" for run in RUNS]
MODELS = [RUN_DIR.format(**run) + "/model" for run in RUNS]
PLOTS = [RUN_DIR.format(**run) + f"/{kind}.png" for run in RUNS for kind in CFG["plots"]]


def code(*modules):
    return [str(ROOT / "scrna_benchmark" / f"{module}.py") for module in modules]


def lookup(wildcards):
    return LOOKUP[(wildcards.dataset, wildcards.feature, wildcards.model, wildcards.run_id)]


def dump(value):
    return json.dumps(value, sort_keys=True)


def prepare_settings(dataset):
    ds = CFG["datasets"][dataset]
    keys = ("path", "counts_layer", "batch_key", "label_key", "sample_key", "sample_columns",
            "exclude_labels", "gene_symbol_key")
    return {
        "dataset_id": dataset, "dataset": {key: ds[key] for key in keys if key in ds},
        "preprocessing": {key: value for key, value in preprocessing_settings(CFG, dataset).items()
                          if key != "n_top_genes"},
        "require_sample": "mrvi" in CFG["models"],
    }


def evaluation_settings(dataset):
    settings = dict(CFG["evaluation"])
    ds = CFG["datasets"][dataset]
    settings.update(ds.get("evaluation", {}))
    return {"dataset": {key: ds[key] for key in ("batch_key", "label_key", "pseudotime_key")
                        if key in ds}, "evaluation": settings}


wildcard_constraints:
    dataset="|".join(re.escape(name) for name in CFG["selected_datasets"]),
    feature="full|hvg",
    model="scvi|mrvi|ldvae",
    run_id=r"h\d+_z\d+_l\d+(?:_u\d+)?_s\d+",
    kind="umap|tsne"


rule all:
    input:
        OUT + "/summary/results.csv",
        OUT + "/summary/results.xlsx",
        OUT + "/summary/best_runs.csv",
        OUT + "/summary/metric_status.tsv",
        OUT + "/summary/config.json",
        MODELS,
        PLOTS


rule prepare_counts:
    input:
        h5ad=lambda wc: CFG["datasets"][wc.dataset]["path"],
        runner=RUNNER,
        code=code("data", "io")
    output:
        h5ad=PREPARED + "/full.h5ad",
        metadata=PREPARED + "/full.json"
    params:
        settings=lambda wc: dump(prepare_settings(wc.dataset))
    threads: CFG["resources"]["prepare_threads"]
    resources:
        mem_mb=CFG["resources"]["prepare_mem_mb"]
    log: OUT + "/logs/prepare/{dataset}.log"
    conda: ENVIRONMENT
    shell:
        "python {input.runner:q} prepare --input {input.h5ad:q} --output {output.h5ad:q} "
        "--metadata {output.metadata:q} --settings {params.settings:q} --threads {threads} "
        "> {log:q} 2>&1"


rule select_hvg:
    input:
        h5ad=PREPARED + "/full.h5ad",
        metadata=PREPARED + "/full.json",
        runner=RUNNER,
        code=code("data", "io")
    output:
        h5ad=PREPARED + "/hvg.h5ad",
        metadata=PREPARED + "/hvg.json"
    params:
        settings=lambda wc: dump({
            "n_top_genes": preprocessing_settings(CFG, wc.dataset)["n_top_genes"],
            "batch_key": CFG["datasets"][wc.dataset]["batch_key"],
        })
    threads: CFG["resources"]["prepare_threads"]
    resources:
        mem_mb=CFG["resources"]["prepare_mem_mb"]
    log: OUT + "/logs/hvg/{dataset}.log"
    conda: ENVIRONMENT
    shell:
        "python {input.runner:q} hvg --input {input.h5ad:q} --output {output.h5ad:q} "
        "--metadata {output.metadata:q} --settings {params.settings:q} --threads {threads} "
        "> {log:q} 2>&1"


rule train_model:
    input:
        h5ad=PREPARED + "/{feature}.h5ad",
        preprocessing=PREPARED + "/{feature}.json",
        runner=RUNNER,
        code=code("models", "io")
    output:
        model=directory(RUN_DIR + "/model"),
        embedding=RUN_DIR + "/embedding.h5ad",
        metadata=RUN_DIR + "/training.json"
    params:
        settings=lambda wc: dump(run_settings(CFG, lookup(wc)))
    threads: CFG["resources"]["train_threads"]
    resources:
        gpu=1 if CFG["accelerator"] == "gpu" else 0,
        mem_mb=CFG["resources"]["train_mem_mb"]
    log: OUT + "/logs/train/{dataset}/{feature}/{model}/{run_id}.log"
    benchmark: RUN_DIR + "/train_benchmark.tsv"
    conda: ENVIRONMENT
    shell:
        "python {input.runner:q} train --input {input.h5ad:q} --model-dir {output.model:q} "
        "--output {output.embedding:q} --metadata {output.metadata:q} "
        "--settings {params.settings:q} --threads {threads} > {log:q} 2>&1"


rule evaluate_model:
    input:
        reference=PREPARED + "/full.h5ad",
        embedding=RUN_DIR + "/embedding.h5ad",
        training=RUN_DIR + "/training.json",
        runner=RUNNER,
        code=code("evaluation", "metrics", "io")
    output:
        metrics=RUN_DIR + "/metrics.json",
        evaluated=RUN_DIR + "/evaluated.h5ad"
    params:
        settings=lambda wc: dump(evaluation_settings(wc.dataset))
    threads: CFG["resources"]["evaluate_threads"]
    resources:
        mem_mb=CFG["resources"]["evaluate_mem_mb"]
    log: OUT + "/logs/evaluate/{dataset}/{feature}/{model}/{run_id}.log"
    conda: ENVIRONMENT
    shell:
        "python {input.runner:q} evaluate --reference {input.reference:q} "
        "--input {input.embedding:q} --training {input.training:q} --output {output.metrics:q} "
        "--evaluated {output.evaluated:q} --settings {params.settings:q} --threads {threads} "
        "> {log:q} 2>&1"


rule plot_embedding:
    input:
        evaluated=RUN_DIR + "/evaluated.h5ad",
        runner=RUNNER,
        code=code("reporting", "metrics", "io")
    output:
        RUN_DIR + "/{kind}.png"
    params:
        settings=lambda wc: dump({"dataset": {
            key: CFG["datasets"][wc.dataset][key] for key in ("batch_key", "label_key")
        }})
    threads: CFG["resources"]["evaluate_threads"]
    resources:
        mem_mb=CFG["resources"]["evaluate_mem_mb"]
    log: OUT + "/logs/plot/{dataset}/{feature}/{model}/{run_id}_{kind}.log"
    conda: ENVIRONMENT
    shell:
        "python {input.runner:q} plot --input {input.evaluated:q} --output {output:q} "
        "--kind {wildcards.kind:q} --settings {params.settings:q} --threads {threads} "
        "> {log:q} 2>&1"


rule aggregate_results:
    input:
        metrics=METRICS,
        runner=RUNNER,
        code=code("reporting", "metrics", "io")
    output:
        csv=OUT + "/summary/results.csv",
        xlsx=OUT + "/summary/results.xlsx",
        best=OUT + "/summary/best_runs.csv",
        status=OUT + "/summary/metric_status.tsv",
        config=OUT + "/summary/config.json"
    params:
        settings=lambda wc: dump(CFG)
    threads: 1
    resources:
        mem_mb=2048
    log: OUT + "/logs/aggregate.log"
    conda: ENVIRONMENT
    shell:
        "python {input.runner:q} aggregate --metrics {input.metrics:q} "
        "--csv {output.csv:q} --xlsx {output.xlsx:q} --best {output.best:q} "
        "--status {output.status:q} --config-output {output.config:q} "
        "--settings {params.settings:q} > {log:q} 2>&1"
