"""Command-line stages used by the root Snakemake workflow."""

import argparse
import json
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "hvg", "train", "evaluate", "plot", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--settings", type=json.loads, required=True)
        command.add_argument("--threads", type=int, default=1)
        if name != "aggregate":
            command.add_argument("--input", required=True)
            command.add_argument("--output", required=True)
        if name in ("prepare", "hvg", "train"):
            command.add_argument("--metadata", required=True)
        if name == "train":
            command.add_argument("--model-dir", required=True)
        if name == "evaluate":
            command.add_argument("--reference", required=True)
            command.add_argument("--training", required=True)
            command.add_argument("--evaluated", required=True)
        if name == "plot":
            command.add_argument("--kind", choices=("umap", "tsne"), required=True)
        if name == "aggregate":
            command.add_argument("--metrics", nargs="+", required=True)
            for output in ("csv", "xlsx", "best", "status", "config-output"):
                command.add_argument(f"--{output}", required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    # Set thread limits before importing numerical libraries in any stage.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.setdefault("MPLBACKEND", "Agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "prepare":
        from scrna_benchmark.data import prepare_counts
        prepare_counts(args.input, args.output, args.metadata, args.settings)
    elif args.stage == "hvg":
        from scrna_benchmark.data import select_hvg
        select_hvg(args.input, args.output, args.metadata, args.settings)
    elif args.stage == "train":
        from scrna_benchmark.models import train_model
        train_model(args.input, args.model_dir, args.output, args.metadata, args.settings)
    elif args.stage == "evaluate":
        from scrna_benchmark.evaluation import evaluate
        evaluate(args.reference, args.input, args.training, args.output, args.evaluated,
                 args.settings, args.threads)
    elif args.stage == "plot":
        from scrna_benchmark.reporting import plot_embedding
        plot_embedding(args.input, args.output, args.kind, args.settings)
    elif args.stage == "aggregate":
        from scrna_benchmark.reporting import aggregate
        aggregate(args.metrics, args.csv, args.xlsx, args.best, args.status,
                  args.config_output, args.settings)


if __name__ == "__main__":
    main()
