"""
ASR Benchmark Analysis Script

Reads per-model CSVs from results/raw/, produces:
  - results/aggregated/all_results.csv
  - results/aggregated/summary_by_model.csv
  - results/aggregated/summary_by_condition.csv
  - results/aggregated/failure_cases.csv
  - results/aggregated/wer_by_model_boxplot.png
  - results/aggregated/wer_by_condition_grouped_bar.png
  - results/aggregated/entity_accuracy_bar.png
  - results/aggregated/latency_boxplot.png
  - Markdown pivot tables printed to stdout

Usage:
  python scripts/analyze.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works on Colab and headless
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "results" / "raw"
AGG_DIR = ROOT / "results" / "aggregated"
MODEL_NAMES = ["deepgram", "whisper", "ai4bharat"]
COLORS = {"deepgram": "#4C72B0", "whisper": "#DD8452", "ai4bharat": "#55A868"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    frames = []
    for name in MODEL_NAMES:
        csv = RAW_DIR / f"results_{name}.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
        else:
            print(f"[WARN] Missing: {csv.name} — skipping")
    if not frames:
        print("No result CSVs found. Run pipeline.py first.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    # Coerce numeric columns; error rows get NaN wer/cer
    for col in ("wer", "cer", "latency_ms", "fuzzy_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["entity_hit"] = pd.to_numeric(df["entity_hit"], errors="coerce").fillna(0).astype(int)
    return df


# ── Aggregated tables ─────────────────────────────────────────────────────────

def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["error"].fillna("") == ""]
    grp = valid.groupby("model")
    summary = pd.DataFrame({
        "n_samples": grp["filename"].count(),
        "n_errors": df.groupby("model")["error"].apply(lambda s: (s.fillna("") != "").sum()),
        "mean_wer": grp["wer"].mean().round(4),
        "median_wer": grp["wer"].median().round(4),
        "mean_cer": grp["cer"].mean().round(4),
        "entity_accuracy": grp["entity_hit"].mean().round(4),
        "mean_latency_ms": grp["latency_ms"].mean().round(1),
        "p95_latency_ms": grp["latency_ms"].quantile(0.95).round(1),
    }).reset_index()
    return summary


def summary_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["error"].fillna("") == ""]
    grp = valid.groupby(["model", "condition"])
    summary = pd.DataFrame({
        "n_samples": grp["filename"].count(),
        "mean_wer": grp["wer"].mean().round(4),
        "entity_accuracy": grp["entity_hit"].mean().round(4),
        "mean_latency_ms": grp["latency_ms"].mean().round(1),
    }).reset_index()
    return summary


def failure_cases(df: pd.DataFrame) -> pd.DataFrame:
    fails = df[df["entity_hit"] == 0].copy()
    fails = fails.sort_values("wer", ascending=False)
    fails["wer_rank"] = range(1, len(fails) + 1)
    return fails


# ── Charts ────────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved chart → {path.relative_to(ROOT)}")


def chart_wer_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["error"].fillna("") == ""]
    models = [m for m in MODEL_NAMES if m in valid["model"].unique()]
    data = [valid[valid["model"] == m]["wer"].dropna().tolist() for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS.get(model, "#999999"))
    ax.set_title("WER Distribution by Model")
    ax.set_ylabel("Word Error Rate")
    ax.set_xlabel("Model")
    _savefig(fig, out_dir / "wer_by_model_boxplot.png")


def chart_wer_by_condition(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["error"].fillna("") == ""]
    cond_order = ["quiet", "noisy", "phone", "whispered"]
    pivot = valid.pivot_table(values="wer", index="condition", columns="model", aggfunc="mean")
    pivot = pivot.reindex([c for c in cond_order if c in pivot.index])
    models = [m for m in MODEL_NAMES if m in pivot.columns]
    pivot = pivot[models]

    n_conditions = len(pivot)
    n_models = len(models)
    x = range(n_conditions)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        offsets = [xi + i * bar_w for xi in x]
        ax.bar(offsets, pivot[model], width=bar_w, label=model, color=COLORS.get(model, "#999999"))

    ax.set_xticks([xi + bar_w * (n_models - 1) / 2 for xi in x])
    ax.set_xticklabels(pivot.index)
    ax.set_title("Mean WER by Condition and Model")
    ax.set_ylabel("Mean WER")
    ax.set_xlabel("Condition")
    ax.legend()
    _savefig(fig, out_dir / "wer_by_condition_grouped_bar.png")


def chart_entity_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["error"].fillna("") == ""]
    acc = valid.groupby("model")["entity_hit"].mean()
    models = [m for m in MODEL_NAMES if m in acc.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, [acc[m] for m in models], color=[COLORS.get(m, "#999999") for m in models])
    ax.set_title("Entity-Level Locality Accuracy by Model")
    ax.set_ylabel("Accuracy (fraction of samples)")
    ax.set_ylim(0, 1.05)
    for i, m in enumerate(models):
        ax.text(i, acc[m] + 0.02, f"{acc[m]:.2f}", ha="center", fontsize=10)
    _savefig(fig, out_dir / "entity_accuracy_bar.png")


def chart_latency_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["error"].fillna("") == ""]
    models = [m for m in MODEL_NAMES if m in valid["model"].unique()]
    data = [valid[valid["model"] == m]["latency_ms"].dropna().tolist() for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS.get(model, "#999999"))
    ax.set_title("Latency Distribution by Model")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Model")
    _savefig(fig, out_dir / "latency_boxplot.png")


# ── Markdown pivot tables ─────────────────────────────────────────────────────

def print_markdown_summary(by_model: pd.DataFrame, by_condition: pd.DataFrame) -> None:
    print("\n=== BENCHMARK SUMMARY ===\n")

    print("**By Model:**")
    print(by_model.to_markdown(index=False))

    print("\n**WER Pivot (model × condition):**")
    valid_cols = [c for c in ["model", "condition", "mean_wer"] if c in by_condition.columns]
    pivot = by_condition[valid_cols].pivot(index="model", columns="condition", values="mean_wer")
    cond_order = [c for c in ["quiet", "noisy", "phone", "whispered"] if c in pivot.columns]
    pivot = pivot[cond_order]
    print(pivot.to_markdown())

    print("\n**Entity Accuracy Pivot (model × condition):**")
    valid_cols2 = [c for c in ["model", "condition", "entity_accuracy"] if c in by_condition.columns]
    pivot2 = by_condition[valid_cols2].pivot(index="model", columns="condition", values="entity_accuracy")
    pivot2 = pivot2[[c for c in cond_order if c in pivot2.columns]]
    print(pivot2.to_markdown())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results()

    all_path = AGG_DIR / "all_results.csv"
    df.to_csv(all_path, index=False)
    print(f"  Merged CSV → {all_path.relative_to(ROOT)}  ({len(df)} rows)")

    by_model = summary_by_model(df)
    by_model.to_csv(AGG_DIR / "summary_by_model.csv", index=False)

    by_cond = summary_by_condition(df)
    by_cond.to_csv(AGG_DIR / "summary_by_condition.csv", index=False)

    fails = failure_cases(df)
    fails.to_csv(AGG_DIR / "failure_cases.csv", index=False)
    print(f"  Failure cases: {len(fails)} rows")

    print("\nGenerating charts...")
    chart_wer_boxplot(df, AGG_DIR)
    chart_wer_by_condition(df, AGG_DIR)
    chart_entity_accuracy(df, AGG_DIR)
    chart_latency_boxplot(df, AGG_DIR)

    print_markdown_summary(by_model, by_cond)


if __name__ == "__main__":
    main()
