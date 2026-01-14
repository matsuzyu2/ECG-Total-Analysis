#!/usr/bin/env python3
"""Compare Good vs Non-Good responders by intervention effect magnitude.

This script scans per-session outputs created by `run_resting_hr_feedback_analysis.py`:
- Results/<session_id>/resting_hr_feedback_metrics.csv

For each session/subject, it computes median-based deltas:
- Control_Delta = Control_Post - Control_Pre
- Target_Delta  = Target_Post  - Target_Pre
- Diff          = Target_Delta - Control_Delta

Classification follows `report_good_responders.py`:
- BF_Type == 'Dec': Diff <= -threshold
- BF_Type == 'Inc': Diff >=  threshold

Then it plots a direction-aligned intervention effect:
- Plot Diff = Diff (Inc) / -Diff (Dec)

Responder criteria is unchanged; only the visualization metric is sign-aligned
for the Decrease condition.

Output:
- Results/good_vs_non_good_comparison.svg

Usage:
  python src/plot_good_vs_non_good.py
  python src/plot_good_vs_non_good.py --threshold 1.0
  python src/plot_good_vs_non_good.py --output Results/good_vs_non_good_comparison.svg
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config


@dataclass(frozen=True)
class SessionDeltas:
    subject_id: str
    session_id: str
    bf_type: str
    control_delta: float
    target_delta: float
    diff: float


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
    """Return Median HR if available, else fall back to Mean HR."""

    row = metrics_df[(metrics_df["condition"] == condition) & (metrics_df["phase"] == phase)]
    if row.empty:
        return float("nan")

    v = row.iloc[0].get("time_median_hr")
    if pd.notna(v):
        return _safe_float(v)

    return _safe_float(row.iloc[0].get("time_mean_hr"))


def _is_good_responder(*, bf_type: str, diff: float, threshold: float) -> bool:
    if pd.isna(diff):
        return False
    if bf_type == "Dec":
        return diff <= -threshold
    if bf_type == "Inc":
        return diff >= threshold
    return False


def _read_session_deltas(metrics_csv: Path) -> Optional[SessionDeltas]:
    df = pd.read_csv(metrics_csv)

    required = {"condition", "phase", "BF_Type", "Subject"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    if "time_median_hr" not in df.columns and "time_mean_hr" not in df.columns:
        raise ValueError(
            f"Missing HR column(s) in {metrics_csv}: needs time_median_hr or time_mean_hr"
        )

    bf_type = str(df["BF_Type"].dropna().iloc[0]) if df["BF_Type"].notna().any() else ""
    subject_id = str(df["Subject"].dropna().iloc[0]) if df["Subject"].notna().any() else ""
    session_id = metrics_csv.parent.name

    if not subject_id:
        return None

    # Only judge Inc/Dec.
    if bf_type not in {"Inc", "Dec"}:
        return None

    control_pre = _get_primary_hr(df, condition="control", phase="pre")
    control_post = _get_primary_hr(df, condition="control", phase="post")
    target_pre = _get_primary_hr(df, condition="target", phase="pre")
    target_post = _get_primary_hr(df, condition="target", phase="post")

    control_delta = control_post - control_pre if pd.notna(control_pre) and pd.notna(control_post) else float("nan")
    target_delta = target_post - target_pre if pd.notna(target_pre) and pd.notna(target_post) else float("nan")
    diff = target_delta - control_delta if pd.notna(target_delta) and pd.notna(control_delta) else float("nan")

    return SessionDeltas(
        subject_id=subject_id,
        session_id=session_id,
        bf_type=bf_type,
        control_delta=float(control_delta),
        target_delta=float(target_delta),
        diff=float(diff),
    )


def build_dataset(*, results_dir: Path, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    skipped = 0

    for metrics_csv in _iter_session_metrics_csvs(results_dir):
        try:
            d = _read_session_deltas(metrics_csv)
        except Exception:
            skipped += 1
            continue

        if d is None:
            continue

        is_good = _is_good_responder(bf_type=d.bf_type, diff=d.diff, threshold=threshold)

        diff_plot = float("nan")
        if pd.notna(d.diff):
            # Align direction across BF types so that "larger is better" for both:
            # - Inc: positive diff indicates improvement (keep as-is)
            # - Dec: negative diff indicates improvement (flip sign)
            diff_plot = -float(d.diff) if d.bf_type == "Dec" else float(d.diff)

        rows.append(
            {
                "Subject": d.subject_id,
                "session_id": d.session_id,
                "BF_Type": d.bf_type,
                "control_delta": d.control_delta,
                "target_delta": d.target_delta,
                "diff": d.diff,
                "diff_plot": diff_plot,
                # Alias to match terminology in documentation/specs.
                "plot_diff": diff_plot,
                "group": "Good Responders" if is_good else "Non-Good Responders",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No usable metrics found under: {results_dir}/<session>/resting_hr_feedback_metrics.csv")

    out["diff_plot"] = pd.to_numeric(out["diff_plot"], errors="coerce")

    # Keep numeric columns numeric.
    for c in ["control_delta", "target_delta", "diff"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out.attrs["skipped_files"] = skipped
    return out


def _exclude_outliers_mean_2sd_by_group(
    df: pd.DataFrame,
    *,
    value_col: str = "plot_diff",
    group_col: str = "group",
    n_sd: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude outliers using mean ± n_sd*SD, computed within each group.

    Notes
    -----
    - Outlier detection is performed on `value_col`.
    - Mean/SD are computed per `group_col`.
    - Rows with NaN in `value_col` are dropped from the returned clean dataset.
    """

    if df.empty:
        return df.copy(), df.iloc[0:0].copy()

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col, group_col])

    grp = work.groupby(group_col, dropna=False)[value_col]
    mean_s = grp.transform("mean")
    sd_s = grp.transform("std")

    # If SD is not finite/positive, do not exclude anything for that row.
    sd_ok = np.isfinite(sd_s.to_numpy(dtype=float)) & (sd_s.to_numpy(dtype=float) > 0)

    lo = mean_s - float(n_sd) * sd_s
    hi = mean_s + float(n_sd) * sd_s

    inlier = ((work[value_col] >= lo) & (work[value_col] <= hi)) | (~sd_ok)

    clean = work.loc[inlier].copy()
    excluded = work.loc[~inlier].copy()

    if not excluded.empty:
        excluded["_outlier_rule"] = f"mean±{n_sd:g}SD"
        excluded["_outlier_group_mean"] = mean_s.loc[excluded.index].to_numpy(dtype=float)
        excluded["_outlier_group_sd"] = sd_s.loc[excluded.index].to_numpy(dtype=float)
        excluded["_outlier_lo"] = lo.loc[excluded.index].to_numpy(dtype=float)
        excluded["_outlier_hi"] = hi.loc[excluded.index].to_numpy(dtype=float)
        excluded = excluded.reset_index(drop=True)

    return clean, excluded


def _students_ttest_ind(*, good: pd.Series, non_good: pd.Series) -> tuple[float, float]:
    """Student's t-test (independent samples), equal variances assumed.

    Returns
    -------
    (t_stat, p_value)
        Returns (NaN, NaN) if the test cannot be performed.
    """

    x = pd.to_numeric(good, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(non_good, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    try:
        res = stats.ttest_ind(x, y, equal_var=True, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _format_p_value(p: float) -> str:
    """Format p-value for display."""

    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _p_to_stars(p: float) -> str:
    """Return significance stars only (for figure annotations)."""

    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _draw_pvalue_bracket(*, ax: plt.Axes, x0: float, x1: float, y: float, h: float, text: str) -> None:
    """Draw a bracket between two categories with a label above."""

    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="#111111", linewidth=1.2, clip_on=False)
    ax.text((x0 + x1) / 2.0, y + h, text, ha="center", va="bottom", fontsize=10, color="#111111")


def plot_good_vs_non_good(*, df: pd.DataFrame, threshold: float, output_svg: Path) -> None:
    # Use Arial everywhere (including mathtext).
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )

    sns.set_theme(style="whitegrid")

    order = ["Non-Good Responders", "Good Responders"]
    df = df.copy()
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)

    # Slightly taller aspect to improve readability.
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=180)

    # Color by group: natural gray tones with better contrast.
    palette = {"Non-Good Responders": "#AAAAAA", "Good Responders": "#666666"}

    # Boxplot: x=Group, y=Diff value.
    sns.boxplot(
        data=df,
        x="group",
        y="diff_plot",
        order=order,
        hue="group",
        palette=palette,
        dodge=False,
        width=0.55,
        fliersize=0,
        linewidth=1.1,
        ax=ax,
    )

    # Hue is used only to satisfy seaborn's palette API; remove legend.
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # Overlay individual points.
    sns.stripplot(
        data=df,
        x="group",
        y="diff_plot",
        order=order,
        jitter=True,
        size=3.8,
        color="#000000",
        alpha=0.85,
        ax=ax,
    )

    # Remove legend (requested).
    leg2 = ax.get_legend()
    if leg2 is not None:
        leg2.remove()

    n_good = int((df["group"] == "Good Responders").sum())
    n_non = int((df["group"] == "Non-Good Responders").sum())

    ax.set_xlabel("Group")
    ax.set_ylabel("Diff (Median HR) [bpm] (Dec sign-flipped)")

    # Statistical test (unpaired): Good vs Non-Good on plot metric.
    good_vals = df.loc[df["group"] == "Good Responders", "diff_plot"]
    non_vals = df.loc[df["group"] == "Non-Good Responders", "diff_plot"]
    _t_stat, p = _students_ttest_ind(good=good_vals, non_good=non_vals)
    p_stars = _p_to_stars(p)

    # No title (restore previous style).

    # Symmetric y-limits around 0 for signed metric, with extra headroom
    # so the significance bracket doesn't overlap the boxes.
    y = pd.to_numeric(df["diff_plot"], errors="coerce").dropna()
    if not y.empty:
        y_abs = float(np.max(np.abs(y)))
        y_lim = max(1.0, y_abs * 1.15)
        # Initial ylim
        ax.set_ylim(-y_lim, y_lim)

    # Keep only the 0 reference line on y (requested).
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)

    # Significance bracket above both boxes (ensure no overlap).
    y_top = float(ax.get_ylim()[1])
    y_bottom = float(ax.get_ylim()[0])
    y_span = y_top - y_bottom

    # Place the bracket above the max observed value with a margin.
    y_max = float(y.max()) if not y.empty else 0.0
    margin = 0.06 * y_span
    bracket_y = y_max + margin
    bracket_h = 0.03 * y_span

    # Ensure enough top space; keep symmetry around 0.
    needed_top = bracket_y + bracket_h + 0.04 * y_span
    if needed_top > y_top:
        new_lim = float(needed_top)
        ax.set_ylim(-new_lim, new_lim)
        y_top = float(ax.get_ylim()[1])
        y_bottom = float(ax.get_ylim()[0])
        y_span = y_top - y_bottom
        margin = 0.06 * y_span
        bracket_y = y_max + margin
        bracket_h = 0.03 * y_span

    # Show stars on the bar; keep numeric p out of the figure (restore previous style).
    _draw_pvalue_bracket(ax=ax, x0=0.0, x1=1.0, y=bracket_y, h=bracket_h, text=p_stars)

    # Axis labels:
    # - remove bottom 'Group' label
    # - x-axis label should be 'Diff value'
    ax.set_xlabel("")
    ax.set_ylabel("Diff value (bpm)")

    # L-shaped axes: show only left and bottom spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.tick_params(axis="both", colors="#111111")

    # White background; remove gridlines (keep only the 0-line).
    ax.set_facecolor("#FFFFFF")
    ax.grid(False)

    # Figure-level note (outside plot, bottom-left): italicize only p.
    fig.text(
        0.02,
        0.02,
        r"** $\it{p}$ < .01",
        ha="left",
        va="bottom",
        fontsize=10,
        color="#111111",
    )

    # Keep some margin for the figure note.
    fig.subplots_adjust(bottom=0.14, left=0.12)

    plt.tight_layout()

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Good vs Non-Good responders by |Diff| based on resting HR feedback metrics (Median HR)."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Config().get_results_dir(),
        help="Results directory (default: project Results/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for responder decision (default: 0.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path (default: Results/good_vs_non_good_comparison.svg)",
    )

    args = parser.parse_args(argv)
    results_dir: Path = args.results_dir
    threshold = abs(float(args.threshold))
    output_svg: Path = args.output if args.output is not None else results_dir / "good_vs_non_good_comparison.svg"

    df_raw = build_dataset(results_dir=results_dir, threshold=threshold)

    # Drop rows with missing plot_diff (cannot be used for outlier rejection / t-test / plotting).
    df_valid = df_raw.copy()
    df_valid["plot_diff"] = pd.to_numeric(df_valid["plot_diff"], errors="coerce")
    missing_mask = df_valid["plot_diff"].isna()
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        print("\n" + "=" * 70)
        print("[WARN] Dropping rows with missing plot_diff (cannot plot/test)")
        print("[WARN] Format: Subject (session_id): plot_diff [bpm] {group}")
        for _, r in df_valid.loc[missing_mask].iterrows():
            subj = str(r.get("Subject", ""))
            sid = str(r.get("session_id", ""))
            grp = str(r.get("group", ""))
            print(f"  - {subj} ({sid}): nan {grp}")
        print("=" * 70 + "\n")

    df_valid = df_valid.loc[~missing_mask].copy()

    # Outlier rejection (mean ± 2SD) within each group on plot_diff.
    df, excluded = _exclude_outliers_mean_2sd_by_group(df_valid, value_col="plot_diff", group_col="group", n_sd=2.0)
    df.attrs.update(df_valid.attrs)

    n_excluded = int(len(excluded))
    if n_excluded > 0:
        excluded = excluded.copy()
        excluded["plot_diff"] = pd.to_numeric(excluded["plot_diff"], errors="coerce")
        excluded = excluded.sort_values(["group", "plot_diff"], ascending=[True, False])

        print("\n" + "=" * 70)
        print("[INFO] Excluded outliers (> 2SD within-group; based on plot_diff)")
        print("[INFO] Format: Subject (session_id): plot_diff [bpm] {group}")
        for _, r in excluded.iterrows():
            subj = str(r.get("Subject", ""))
            sid = str(r.get("session_id", ""))
            val = r.get("plot_diff")
            grp = str(r.get("group", ""))
            try:
                val_txt = f"{float(val):.2f}"
            except Exception:
                val_txt = "nan"
            print(f"  - {subj} ({sid}): {val_txt} {grp}")
        print("=" * 70 + "\n")

    # Calculate and print statistics (AFTER outlier exclusion)
    print("\n=== Statistical Analysis (After Outlier Exclusion) ===")
    print(f"Total subjects scanned: {len(df_raw)}")
    print(f"Valid plot_diff: {len(df_valid)} (dropped_missing={n_missing})")
    print(f"Excluded outliers: {n_excluded} (rule: mean±2SD within each group)")
    print(f"Remaining subjects: {len(df)}")

    good_mask = df["group"] == "Good Responders"
    non_good_mask = df["group"] == "Non-Good Responders"

    n_good = int(good_mask.sum())
    n_non = int(non_good_mask.sum())

    print(f"Good Responders: {n_good}")
    print(f"Non-Good Responders: {n_non}")
    
    # Breakdown by BF_Type
    print("\n--- Breakdown by Intervention Type ---")
    for bf_type in ["Inc", "Dec"]:
        df_type = df[df["BF_Type"] == bf_type]
        if df_type.empty:
            continue
        n_good_type = int((df_type["group"] == "Good Responders").sum())
        n_non_type = int((df_type["group"] == "Non-Good Responders").sum())
        print(f"{bf_type}: Good={n_good_type}, Non-Good={n_non_type}")
    
    # Descriptive statistics for plot_diff
    print("\n--- Descriptive Statistics (plot_diff [bpm]) ---")
    good_vals = df.loc[good_mask, "plot_diff"].dropna()
    non_vals = df.loc[non_good_mask, "plot_diff"].dropna()
    
    if not good_vals.empty:
        print(f"Good Responders:")
        print(f"  Mean: {good_vals.mean():.2f}")
        print(f"  Median: {good_vals.median():.2f}")
        print(f"  SD: {good_vals.std():.2f}")
        print(f"  Range: [{good_vals.min():.2f}, {good_vals.max():.2f}]")
    
    if not non_vals.empty:
        print(f"Non-Good Responders:")
        print(f"  Mean: {non_vals.mean():.2f}")
        print(f"  Median: {non_vals.median():.2f}")
        print(f"  SD: {non_vals.std():.2f}")
        print(f"  Range: [{non_vals.min():.2f}, {non_vals.max():.2f}]")
    
    # Student's t-test
    print("\n--- Student's t-test (ttest_ind, equal_var=True) ---")
    t_stat, p = _students_ttest_ind(good=good_vals, non_good=non_vals)
    p_text = _format_p_value(p)

    if np.isfinite(p):
        print(f"t statistic: {t_stat:.3f}")
        print(f"P-value: {p:.4f}")
        print(f"Significance: {p_text}")
    else:
        print("Could not perform Student's t-test (insufficient data)")
    
    print("\n" + "="*40 + "\n")
    
    plot_good_vs_non_good(df=df, threshold=threshold, output_svg=output_svg)

    skipped = int(df.attrs.get("skipped_files", 0))
    print(
        f"Saved: {output_svg} (valid={len(df_valid)}, n={len(df)} after exclusion, "
        f"excluded={n_excluded}, dropped_missing={n_missing}, skipped_files={skipped})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
