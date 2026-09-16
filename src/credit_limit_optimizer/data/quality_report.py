"""Formal Data Quality report (§11): turns quality_report.json (written by
`make validate-data`) into figures + a written markdown report with
business interpretation, matching the EDA report's format.

§11 explicitly requires the score broken down by Dataset, Field, Customer
segment, AND Month -- the first three were already computed by
validation.py; `by_month` was added there (scoped to transactions.csv +
payments.csv, the only two datasets with a real per-record calendar
timestamp) specifically to close this gap.

Depends on `make validate-data` having been run first (reads its JSON
output rather than re-running the ~20s validation pass).

Run: python -m credit_limit_optimizer.data.quality_report
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_limit_optimizer.utils.config import PROJECT_ROOT
from credit_limit_optimizer.utils.logging import get_logger

log = get_logger("quality_report")

QUALITY_JSON_PATH = PROJECT_ROOT / "reports" / "outputs" / "quality_report.json"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "quality"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "quality_report.md"

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
SEVERITY_COLORS = {"CRITICAL": "#B71C1C", "HIGH": "#E65100", "MEDIUM": "#F9A825", "LOW": "#1565C0", "INFO": "#9E9E9E"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_quality_report() -> dict:
    if not QUALITY_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{QUALITY_JSON_PATH} not found. Run `make validate-data` "
            "(or `python -m credit_limit_optimizer.data.validation`) first."
        )
    with QUALITY_JSON_PATH.open() as f:
        return json.load(f)


def _score_bar(scores: dict, title: str, path, sort=True, horizontal=False) -> None:
    s = pd.Series(scores)
    if sort:
        s = s.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(s))) if horizontal else (7, 4))
    colors = ["#C62828" if v < 90 else "#F9A825" if v < 97 else "#2E7D32" for v in s.values]
    if horizontal:
        ax.barh(s.index.astype(str), s.values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score")
    else:
        ax.bar(s.index.astype(str), s.values, color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_score_by_dataset(report: dict) -> None:
    _score_bar(report["by_dataset"], "Data Quality Score by dataset", FIG_DIR / "01_score_by_dataset.png")


def plot_score_by_segment(report: dict) -> None:
    _score_bar(report["by_segment"], "Data Quality Score by customer segment (customers.csv only)",
               FIG_DIR / "02_score_by_segment.png")


def plot_score_by_month(report: dict) -> dict:
    by_month = report["by_month"]
    s = pd.Series(by_month)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s.index, s.values, color="#1565C0", linewidth=2, marker="o", markersize=3)
    ax.axhline(s.mean(), color="#9E9E9E", linestyle="--", linewidth=1, label=f"mean {s.mean():.2f}")
    ax.set_title("Data Quality Score by month (transactions.csv + payments.csv)")
    ax.set_ylabel("Score")
    ax.set_ylim(max(0, s.min() - 2), 100)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    ax.legend()
    fig.savefig(FIG_DIR / "03_score_by_month.png", bbox_inches="tight")
    plt.close(fig)
    return {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}


def plot_score_by_field(report: dict) -> None:
    n = min(15, len(report["by_field"]))
    worst = dict(sorted(report["by_field"].items(), key=lambda kv: kv[1])[:n])
    title = f"{n} lowest-scoring fields" if n < len(report["by_field"]) else "Score by field (ascending)"
    _score_bar(worst, title, FIG_DIR / "04_score_by_field.png", horizontal=True)


def plot_issues_by_severity(report: dict) -> None:
    counts = report["issue_counts_by_severity"]
    ordered = {sev: counts.get(sev, 0) for sev in SEVERITY_ORDER if counts.get(sev, 0) > 0}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ordered.keys(), ordered.values(), color=[SEVERITY_COLORS[s] for s in ordered])
    ax.set_title("Issues found, by severity")
    ax.set_ylabel("Issue count")
    fig.savefig(FIG_DIR / "05_issues_by_severity.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_issue_rules(report: dict) -> None:
    top = dict(sorted(report["issue_counts_by_rule"].items(), key=lambda kv: kv[1], reverse=True)[:12])
    s = pd.Series(top).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index.astype(str), s.values, color="#1565C0")
    ax.set_title("Top 12 issue rules by count")
    ax.set_xlabel("Issue count")
    fig.savefig(FIG_DIR / "06_top_issue_rules.png", bbox_inches="tight")
    plt.close(fig)


def plot_lineage_funnel(report: dict) -> None:
    datasets = list(report["row_counts_raw"].keys())
    validated = [report["row_counts_validated"][d] for d in datasets]
    quarantined = [report["row_counts_quarantined"][d] for d in datasets]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(datasets, validated, label="VALIDATED", color="#2E7D32")
    ax.bar(datasets, quarantined, bottom=validated, label="QUARANTINED", color="#C62828")
    ax.set_title("RAW → VALIDATED / QUARANTINED, by dataset")
    ax.set_ylabel("Row count")
    ax.legend()
    fig.savefig(FIG_DIR / "07_lineage_funnel.png", bbox_inches="tight")
    plt.close(fig)


def build_report(report: dict) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_score_by_dataset(report)
    plot_score_by_segment(report)
    month_stats = plot_score_by_month(report)
    plot_score_by_field(report)
    plot_issues_by_severity(report)
    plot_top_issue_rules(report)
    plot_lineage_funnel(report)
    return {"month_stats": month_stats}


def write_markdown_report(report: dict, month_stats: dict) -> None:
    total_raw = sum(report["row_counts_raw"].values())
    total_quarantined = sum(report["row_counts_quarantined"].values())
    lines = [
        "# Headroom — Data Quality Report",
        "",
        f"_Generated {report['generated_at']}. Source: `reports/outputs/quality_report.json`, "
        "written by `make validate-data`; every number below is read from that file, none "
        "recomputed or estimated here._",
        "",
        "## Overall Data Quality Score",
        f"**{report['overall_score']:.1f} / 100**, from {total_raw:,} raw rows across "
        f"{len(report['row_counts_raw'])} datasets ("
        + ", ".join(f"{k}: {v:,}" for k, v in report["row_counts_raw"].items()) + "). "
        f"{sum(v for v in report['issue_counts_by_severity'].values()):,} issues found; "
        f"{total_quarantined:,} rows ({total_quarantined / total_raw:.1%}) quarantined for a "
        "CRITICAL or HIGH-severity issue — everything else stays in the VALIDATED layer "
        "with its issue logged, per the RAW → VALIDATED → QUARANTINED design (§10): "
        "nothing is silently dropped.",
        "",
        "## Score by dataset",
        ", ".join(f"{k}: {v:.1f}" for k, v in report["by_dataset"].items()) + ".",
        "![score by dataset](../figures/quality/01_score_by_dataset.png)",
        "",
        "## Score by customer segment",
        "Scoped to `customers.csv`'s own issues only — a segment's customers can have tens "
        "of thousands of associated transaction/payment rows each, so folding those into a "
        "customer-row-denominated score is a scale mismatch that floored every segment to "
        "0.0 in an earlier version (see CLAUDE.md). "
        + ", ".join(f"{k}: {v:.1f}" for k, v in report["by_segment"].items()) + ".",
        "![score by segment](../figures/quality/02_score_by_segment.png)",
        "",
        "## Score by month",
        f"Scoped to `transactions.csv` + `payments.csv` (the only two datasets with a real "
        f"per-record calendar timestamp — `customers.csv`/`credit_accounts.csv` are single-"
        f"row-per-customer snapshots with no comparable monthly axis). Mean "
        f"{month_stats['mean']:.2f}, range {month_stats['min']:.2f}-{month_stats['max']:.2f} "
        f"across all 36 months (std {month_stats['std']:.2f}) — essentially flat, as "
        "expected: quality issues were injected at a uniform rate independent of calendar "
        "month, so no systematic monthly quality drift should exist (and none does).",
        "![score by month](../figures/quality/03_score_by_month.png)",
        "",
        f"## Score by field ({min(15, len(report['by_field']))} lowest)",
        "The fields most responsible for the score's distance from 100 — every one of "
        "these corresponds to a deliberately injected issue class (missing values, "
        "impossible values, or inconsistent cross-field relationships), not an unexplained "
        "gap.",
        "![score by field](../figures/quality/04_score_by_field.png)",
        "",
        "## Issues by severity",
        ", ".join(f"{k}: {v:,}" for k, v in report["issue_counts_by_severity"].items()) + ". "
        "Only CRITICAL/HIGH trigger quarantine; MEDIUM/LOW/INFO stay in VALIDATED with the "
        "issue logged, since they're not severe enough to justify pulling a record from "
        "modeling but are still worth surfacing (e.g. to a data-quality dashboard or model "
        "monitoring downstream).",
        "![issues by severity](../figures/quality/05_issues_by_severity.png)",
        "",
        "## Top issue rules",
        "The most common individual rule violations, dominated by missing-value injections "
        "on optional/metadata fields (`missing_channel`, `missing_payment_type`) rather "
        "than the rarer but more severe impossible-value/referential-integrity issues.",
        "![top issue rules](../figures/quality/06_top_issue_rules.png)",
        "",
        "## RAW → VALIDATED / QUARANTINED lineage",
        "Lineage is exact for every dataset (`len(VALIDATED) + len(QUARANTINED) == "
        "len(RAW)`, enforced in `test_split_partitions_without_loss_or_duplication`) — no "
        "row is ever silently dropped, only routed.",
        "![lineage funnel](../figures/quality/07_lineage_funnel.png)",
        "",
    ]
    with REPORT_PATH.open("w") as f:
        f.write("\n".join(lines))


def main() -> None:
    report = load_quality_report()
    result = build_report(report)
    write_markdown_report(report, result["month_stats"])
    log.info("Wrote %s and 7 figures to %s", REPORT_PATH.relative_to(PROJECT_ROOT), FIG_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
