"""Metric names and the paper's equally weighted group averages."""

import math


BATCH_METRICS = ("Batch ASW", "PCR Batch", "iLISI", "Graph Connectivity")
BIO_METRICS = ("NMI", "ARI", "Label ASW", "Isolated Label F1", "Isolated Label ASW",
               "cLISI", "Trajectory Conservation")
ALL_METRICS = BATCH_METRICS + BIO_METRICS


def composite_scores(metrics):
    """Exclude explicit N/A metrics, but never hide a failed metric in an average."""
    def group(names):
        values = []
        for name in names:
            result = metrics[name]
            if result["status"] == "error":
                return None, len([n for n in names if metrics[n]["status"] == "ok"])
            if result["status"] == "ok":
                value = result["value"]
                if value is None or not math.isfinite(value):
                    raise ValueError(f"Metric {name} is marked ok but is not finite.")
                values.append(value)
            elif result["status"] not in ("not_applicable", "disabled"):
                raise ValueError(f"Unknown metric status for {name}: {result['status']}")
        return (sum(values) / len(values) if values else None), len(values)

    batch, batch_count = group(BATCH_METRICS)
    biology, bio_count = group(BIO_METRICS)
    overall = (batch + biology) / 2 if batch is not None and biology is not None else None
    return {"Overall": overall, "Overall Batch": batch, "Overall Bio": biology,
            "Batch Metric Count": batch_count, "Bio Metric Count": bio_count}
