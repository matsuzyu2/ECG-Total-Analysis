#!/usr/bin/env python3
"""Plot during-HRFB delta summary (Last5 - Pre5) as a static SVG.

Reads per-session outputs created by `run_during_hrfb_analysis.py`:
- Results/<session_id>/during_hrfb_metrics.csv

Delta definition:
    Delta = MeanHR(Last5) - MeanHR(Pre5)

X-axis categories (ordered):
- Control (Dec group)
- Dec (Target)
- Control (Inc group)
- Inc (Target)

Output:
- Results/during_hrfb_delta_summary.svg
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns

# Ensure scripts work when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass(frozen=True)
class SubjectDeltas:
    subject_id: str
    session_id: str
    bf_type: str  # Inc / Dec
    control_delta: float  # Last5 - First5
    target_delta: float  # Last5 - First5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_session_metrics_csvs(results_dir: Path) -> Iterable[Path]:
    for session_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        csv_path = session_dir / "during_hrfb_metrics.csv"
        if csv_path.exists():
            yield csv_path


def _safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x))
    except Exception:
        return float("nan")


def _get_mean_hr(df: pd.DataFrame, *, condition: str, phase: str) -> float:
    row = df[(df["condition"] == condition) & (df["phase"] == phase)]
    if row.empty:
        return float("nan")
    return _safe_float(row.iloc[0].get("time_mean_hr"))


def _read_subject_deltas(metrics_csv: Path) -> Optional[SubjectDeltas]:
    df = pd.read_csv(metrics_csv)
    required = {"condition", "phase", "time_mean_hr", "BF_Type", "Subject"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    bf_type = str(df["BF_Type"].dropna().iloc[0]) if df["BF_Type"].notna().any() else ""
    subject_id = str(df["Subject"].dropna().iloc[0]) if df["Subject"].notna().any() else ""
    session_id = metrics_csv.parent.name

    control_first = _get_mean_hr(df, condition="Control", phase="Pre5")
    control_last = _get_mean_hr(df, condition="Control", phase="Last5")
    target_first = _get_mean_hr(df, condition="Target", phase="Pre5")
    target_last = _get_mean_hr(df, condition="Target", phase="Last5")

    control_delta = control_last - control_first if pd.notna(control_first) and pd.notna(control_last) else float("nan")
    target_delta = target_last - target_first if pd.notna(target_first) and pd.notna(target_last) else float("nan")

    if not subject_id:
        return None

    return SubjectDeltas(
        subject_id=subject_id,
        session_id=session_id,
        bf_type=bf_type,
        control_delta=control_delta,
        target_delta=target_delta,
    )


def _bf_type_to_group(bf_type: str) -> Optional[str]:
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
            rows.append({"subject_id": d.subject_id, "session_id": d.session_id, "bf_type": d.bf_type, "x": "Con (Dec)", "kind": "Control", "y": d.control_delta})
            rows.append({"subject_id": d.subject_id, "session_id": d.session_id, "bf_type": d.bf_type, "x": "Dec", "kind": "Target", "y": d.target_delta})
        else:
            rows.append({"subject_id": d.subject_id, "session_id": d.session_id, "bf_type": d.bf_type, "x": "Con (Inc)", "kind": "Control", "y": d.control_delta})
            rows.append({"subject_id": d.subject_id, "session_id": d.session_id, "bf_type": d.bf_type, "x": "Inc", "kind": "Target", "y": d.target_delta})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out


def _category_stats(y: pd.Series) -> dict:
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return {"mean": float("nan"), "se": float("nan"), "q1": float("nan"), "median": float("nan"), "q3": float("nan"), "n": 0}
    mean = float(y.mean())
    n = int(y.shape[0])
    se = float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    q1 = float(y.quantile(0.25))
    median = float(y.quantile(0.5))
    q3 = float(y.quantile(0.75))
    return {"mean": mean, "se": se, "q1": q1, "median": median, "q3": q3, "n": n}


def _boost_swarm_spread(*, ax: Axes, cat_index: int, factor: float = 1.8, max_abs: float = 0.42) -> None:
    if not ax.collections:
        return
    coll = ax.collections[-1]
    try:
        offsets = coll.get_offsets()
    except Exception:
        return
    if offsets is None:
        return
    offsets = np.asarray(offsets)
    if offsets.size == 0 or offsets.ndim != 2 or offsets.shape[1] < 2:
        return
    center_x = float(cat_index)
    dx = offsets[:, 0] - center_x
    dx = np.clip(dx * factor, -max_abs, max_abs)
    offsets[:, 0] = center_x + dx
    coll.set_offsets(offsets)


def plot_delta(*, results_dir: Path, output_svg: Path) -> None:
    metrics_csvs = list(_iter_session_metrics_csvs(results_dir))
    deltas: List[SubjectDeltas] = []
    for p in metrics_csvs:
        d = _read_subject_deltas(p)
        if d is not None:
            deltas.append(d)

    long_df = _build_long_df(deltas)

    categories = ["Con (Dec)", "Dec", "Con (Inc)", "Inc"]
    display_labels = ["Control", "Dec", "Control", "Inc"]

    sns.set_theme(style="whitegrid")

    if long_df.empty:
        raise RuntimeError(f"No data found under: {results_dir}/<session>/during_hrfb_metrics.csv")

    long_df = long_df.copy()
    long_df["x"] = pd.Categorical(long_df["x"], categories=categories, ordered=True)

    all_y = pd.to_numeric(long_df["y"], errors="coerce").dropna()
    y_lim = 1.0 if all_y.empty else max(1.0, float(np.max(np.abs(all_y))) * 1.15)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)

    pal = sns.color_palette("colorblind", n_colors=4)
    color_by_cat = dict(zip(categories, pal))

    sns.boxplot(
        data=long_df,
        x="x",
        y="y",
        order=categories,
        color="#F2F2F2",
        width=0.5,
        fliersize=0,
        linewidth=1,
        ax=ax,
    )

    for i, cat in enumerate(categories):
        sub = long_df[long_df["x"] == cat]
        if sub.empty:
            continue
        sns.swarmplot(
            data=sub,
            x="x",
            y="y",
            order=categories,
            color=color_by_cat[cat],
            size=5.0,
            edgecolor="black",
            linewidth=0.3,
            ax=ax,
        )
        _boost_swarm_spread(ax=ax, cat_index=i)

    for i, cat in enumerate(categories):
        sub = long_df[long_df["x"] == cat]
        stats = _category_stats(sub["y"])
        if stats["n"] <= 0:
            continue
        ax.errorbar(
            x=i,
            y=stats["mean"],
            yerr=stats["se"],
            fmt="_",
            color="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=10,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Δ HR (bpm): Last5 − Pre5")
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(display_labels)

    # Legend (control vs target marker proxy)
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666", markeredgecolor="black", markersize=7, label="Control"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#AAA", markeredgecolor="black", markersize=7, label="Target"),
    ]
    ax.legend(handles=legend_items, frameon=False, loc="upper right")

    ax.set_title("During HRFB ΔHR (Last5 − Pre5)")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot during-HRFB delta summary as SVG.")
    parser.add_argument("--results-dir", type=Path, default=_project_root() / "Results")
    parser.add_argument("--output", type=Path, default=_project_root() / "Results" / "during_hrfb_delta_summary.svg")
    args = parser.parse_args()

    plot_delta(results_dir=args.results_dir, output_svg=args.output)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
