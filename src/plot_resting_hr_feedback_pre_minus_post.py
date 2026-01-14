#!/usr/bin/env python3
"""Plot pre-post change summary from already-computed session metrics.

This script DOES NOT rerun the ECG pipeline.
It reads per-session outputs created by `run_resting_hr_feedback_analysis.py`:
- Results/<session_id>/resting_hr_feedback_metrics.csv

It then computes the change amount (post - pre) for median HR (bpm) for both
Control and Target, grouped by BF_Type:
- Dec (Decrease condition)
- Inc (Increase condition)

Output: a single HTML file embedding a swarmplot-style figure.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config


@dataclass(frozen=True)
class SubjectDeltas:
    subject_id: str
    session_id: str
    bf_type: str  # Inc / Dec
    control_delta: float  # post - pre
    target_delta: float  # post - pre


def _iter_session_metrics_csvs(results_dir: Path) -> Iterable[Path]:
    for session_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        csv_path = session_dir / "resting_hr_feedback_metrics.csv"
        if csv_path.exists():
            yield csv_path


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get_primary_hr(metrics_df: pd.DataFrame, *, condition: str, phase: str) -> float:
    row = metrics_df[(metrics_df["condition"] == condition) & (metrics_df["phase"] == phase)]
    if row.empty:
        return float("nan")
    # Prefer robust median HR; fall back to mean HR.
    v = row.iloc[0].get("time_median_hr")
    if pd.notna(v):
        return _safe_float(v)
    return _safe_float(row.iloc[0].get("time_mean_hr"))


def _read_subject_deltas(metrics_csv: Path) -> Optional[SubjectDeltas]:
    df = pd.read_csv(metrics_csv)
    required = {"condition", "phase", "BF_Type", "Subject"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    if "time_median_hr" not in df.columns and "time_mean_hr" not in df.columns:
        raise ValueError(
            f"Missing HR column(s) in {metrics_csv}: needs time_median_hr or time_mean_hr"
        )

    # BF_Type / Subject are duplicated per segment; take the first non-null.
    bf_type = str(df["BF_Type"].dropna().iloc[0]) if df["BF_Type"].notna().any() else ""
    subject_id = str(df["Subject"].dropna().iloc[0]) if df["Subject"].notna().any() else ""
    session_id = metrics_csv.parent.name

    control_pre = _get_primary_hr(df, condition="control", phase="pre")
    control_post = _get_primary_hr(df, condition="control", phase="post")
    target_pre = _get_primary_hr(df, condition="target", phase="pre")
    target_post = _get_primary_hr(df, condition="target", phase="post")

    # Change amount is defined as post - pre.
    control_delta = control_post - control_pre if pd.notna(control_pre) and pd.notna(control_post) else float("nan")
    target_delta = target_post - target_pre if pd.notna(target_pre) and pd.notna(target_post) else float("nan")

    if not subject_id:
        # Can't connect lines without an ID.
        return None

    return SubjectDeltas(
        subject_id=subject_id,
        session_id=session_id,
        bf_type=bf_type,
        control_delta=control_delta,
        target_delta=target_delta,
    )


def _bf_type_to_group(bf_type: str) -> Optional[str]:
    # Expected BF_Type: Inc / Dec
    if bf_type == "Inc":
        return "Inc"
    if bf_type == "Dec":
        return "Dec"
    return None


def _build_long_df(deltas: List[SubjectDeltas]) -> pd.DataFrame:
    rows: List[dict] = []
    for d in deltas:
        group = _bf_type_to_group(d.bf_type)
        if group is None:
            continue

        if group == "Dec":
            rows.append(
                {
                    "subject_id": d.subject_id,
                    "session_id": d.session_id,
                    "bf_type": d.bf_type,
                    "x": "Con (Dec)",
                    "kind": "Control",
                    "y": d.control_delta,
                }
            )
            rows.append(
                {
                    "subject_id": d.subject_id,
                    "session_id": d.session_id,
                    "bf_type": d.bf_type,
                    "x": "Dec",
                    "kind": "Target",
                    "y": d.target_delta,
                }
            )
        elif group == "Inc":
            rows.append(
                {
                    "subject_id": d.subject_id,
                    "session_id": d.session_id,
                    "bf_type": d.bf_type,
                    "x": "Con (Inc)",
                    "kind": "Control",
                    "y": d.control_delta,
                }
            )
            rows.append(
                {
                    "subject_id": d.subject_id,
                    "session_id": d.session_id,
                    "bf_type": d.bf_type,
                    "x": "Inc",
                    "kind": "Target",
                    "y": d.target_delta,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out


def _category_stats(y: pd.Series) -> dict:
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return {
            "mean": float("nan"),
            "se": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "n": 0,
        }
    mean = float(y.mean())
    n = int(y.shape[0])
    se = float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    q1 = float(y.quantile(0.25))
    median = float(y.quantile(0.5))
    q3 = float(y.quantile(0.75))
    return {"mean": mean, "se": se, "q1": q1, "median": median, "q3": q3, "n": n}


def _boost_swarm_spread(*, ax: plt.Axes, cat_index: int, factor: float = 1.8, max_abs: float = 0.42) -> None:
    """Increase horizontal spread of the most recently-added swarm PathCollection.

    seaborn's swarmplot places points to avoid overlaps. This helper scales the
    x-offsets around the category center to make the beeswarm "wider", while
    clipping so points don't bleed into neighboring categories.
    """

    # Find the latest collection with offsets.
    if not ax.collections:
        return
    coll = ax.collections[-1]
    try:
        offsets = coll.get_offsets()
    except Exception:
        return

    if offsets is None or len(offsets) == 0:
        return

    center_x = float(cat_index)
    dx = offsets[:, 0] - center_x
    dx = np.clip(dx * factor, -max_abs, max_abs)
    offsets[:, 0] = center_x + dx
    coll.set_offsets(offsets)


def plot_pre_minus_post(*, results_dir: Path, output_html: Path) -> None:
    metrics_csvs = list(_iter_session_metrics_csvs(results_dir))
    deltas: List[SubjectDeltas] = []
    for p in metrics_csvs:
        d = _read_subject_deltas(p)
        if d is not None:
            deltas.append(d)

    long_df = _build_long_df(deltas)

    categories = ["Con (Dec)", "Dec", "Con (Inc)", "Inc"]
    display_labels = ["Con", "Dec", "Con", "Inc"]

    # Smaller / simpler universal-design plot:
    # - monochrome
    # - swarmplot (beeswarm-like) for sparse points
    # - box for IQR
    # - mean ± SE errorbar
    sns.set_theme(style="whitegrid")

    if long_df.empty:
        raise RuntimeError(f"No data found under: {results_dir}/<session>/resting_hr_feedback_metrics.csv")

    long_df = long_df.copy()
    long_df["x"] = pd.Categorical(long_df["x"], categories=categories, ordered=True)

    all_y = pd.to_numeric(long_df["y"], errors="coerce").dropna()
    y_lim = 1.0 if all_y.empty else max(1.0, float(np.max(np.abs(all_y))) * 1.15)

    # Make the figure narrower as requested.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)

    # Color by condition (universal-design friendly).
    # Uses seaborn's colorblind palette.
    pal = sns.color_palette("colorblind", n_colors=4)
    color_by_cat = dict(zip(categories, pal))

    # 1) Box plot (IQR) - bottom layer
    sns.boxplot(
        data=long_df,
        x="x",
        y="y",
        order=categories,
        color="#F2F2F2",
        width=0.5,
        fliersize=0,
        linewidth=1,
        zorder=1,
        ax=ax,
    )

    # 2) Swarm points - middle layer
    # Shapes: Control=○, Dec(target)=×, Inc(target)=△
    # seaborn swarmplot supports one marker per call; we draw per-category layers
    # so we can control both marker shape and color.
    for cat in categories:
        sub = long_df[long_df["x"] == cat]
        if sub.empty:
            continue

        cat_index = categories.index(cat)

        if cat.startswith("Con"):
            marker = "o"  # Control
            size = 4.5
            lw = 0.4
            ec = "white"
        elif cat == "Dec":
            marker = "X"  # Decrease target
            size = 5.0
            lw = 0.0
            ec = None
        else:  # Inc
            marker = "^"  # Increase target
            size = 5.0
            lw = 0.4
            ec = "white"

        before = len(ax.collections)
        sns.swarmplot(
            data=sub,
            x="x",
            y="y",
            order=categories,
            marker=marker,
            size=size,
            color=color_by_cat[cat],
            linewidth=lw,
            edgecolor=ec,
            zorder=2,
            ax=ax,
        )

        # Ensure the newly-added points are above the box layer.
        for coll in ax.collections[before:]:
            try:
                coll.set_zorder(2)
            except Exception:
                pass

        # Strengthen swarm dispersion.
        # The swarm points are stored as PathCollections; boost the latest one.
        if len(ax.collections) > before:
            _boost_swarm_spread(ax=ax, cat_index=cat_index, factor=1.9, max_abs=0.44)

    # 3) Connect same subject id with a line (within Dec or Inc) - top layer
    # We compute x indices for each category.
    x_index = {c: i for i, c in enumerate(categories)}
    for subject_id, g in long_df.groupby("subject_id"):
        for bf_type, gg in g.groupby("bf_type"):
            order = ["Con (Dec)", "Dec"] if bf_type == "Dec" else ["Con (Inc)", "Inc"]
            gg2 = gg.set_index("x").reindex(order)
            ys = gg2["y"].to_list()
            if any(pd.isna(v) for v in ys):
                continue
            xs = [x_index[o] for o in order]
            ax.plot(xs, ys, color="#999999", linewidth=0.8, zorder=3)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Δ Median HR [bpm]")
    ax.set_title("Resting HR Change by Feedback Condition", fontsize=11)

    # Override x tick labels to match the requested display.
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(display_labels)

    # Restore legend: encode condition by both color and marker shape.
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_by_cat["Con (Dec)"], markeredgecolor="white", markersize=7, label="Con"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=color_by_cat["Dec"], markeredgecolor=color_by_cat["Dec"], markersize=7, label="Dec"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_by_cat["Con (Inc)"], markeredgecolor="white", markersize=7, label="Con"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=color_by_cat["Inc"], markeredgecolor="white", markersize=7, label="Inc"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Condition",
        frameon=False,
        loc="upper left",
    )

    # Make spacing a bit tighter to keep the plot compact.
    plt.tight_layout()

    # Export as PNG in-memory and embed into a single HTML.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "  <title>Resting HR change summary</title>",
            "  <style>body{margin:16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;} img{max-width:100%;height:auto;}</style>",
            "</head>",
            "<body>",
            "  <img alt=\"pre-minus-post summary\" src=\"data:image/png;base64,",
            b64,
            "\" />",
            "</body>",
            "</html>",
        ]
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot post - pre change summary from existing Results/<session>/resting_hr_feedback_metrics.csv outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Config().get_results_dir(),
        help="Results directory (default: project Results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <results-dir>/resting_hr_feedback_post_minus_pre_summary.html)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_html: Path = (
        args.output
        if args.output is not None
        else results_dir / "resting_hr_feedback_post_minus_pre_summary.html"
    )

    plot_pre_minus_post(results_dir=results_dir, output_html=output_html)
    print(f"Saved: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
