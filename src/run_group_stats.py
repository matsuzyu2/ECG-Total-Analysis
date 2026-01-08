#!/usr/bin/env python3
"""Step 3: Group Statistical Analysis & Visualization (Protocol v2)

This script performs group-level statistical analysis and visualization from
HRV summary CSVs produced by Step 2.

Input:
    Results/hrv_summaries/*_hrv_summary.csv

Metadata:
    Data/conditions.csv

Behavior:
    - If Data/conditions.csv does not exist, this script auto-generates a
      template covering all session IDs found in Results/hrv_summaries and exits
      successfully. Fill in the template and re-run.

Protocol v2 summary:
    - Exclude any segment containing "Practice".
    - Phase mapping:
        Resting      : segment contains "Resting"
        GoNoGo_Task  : segment contains "GoNoGo"
        HR_Feedback  : segment contains "Session"
    - Condition mapping:
        Resting:
            Resting_HR_1_Set* => Baseline
            Resting_HR_2_Set* => Control/Target via conditions.csv Set mapping
            Pairing is strict by Set (Baseline within the same Set).
        GoNoGo_Task:
            GoNoGo_Baseline   => Baseline
            GoNoGo_Set*       => Control/Target via Set mapping
            Baseline is re-used for both Set1/Set2 comparisons.
        HR_Feedback:
            Session_Control/Increase/Decrease => explicit condition (priority)
            Else Session_01/Session_02 => Set1/Set2 via Set mapping
            Else => exclude with warning
            No Baseline comparisons; only Control vs Target.

Outputs:
    Results/stats/
        group_stats_results.csv
        slope_{group}_{phase}_{metric}.png
        box_{group}_{phase}_{metric}.png

Usage:
    python src/run_group_stats.py
    python src/run_group_stats.py --quiet

"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config


METRICS: Tuple[str, ...] = (
    "time_mean_hr",
    "time_rmssd",
    "freq_lf_hf_ratio",
)


@dataclass(frozen=True)
class ComparisonSpec:
    phase: str
    group: str
    metric: str
    a: str
    b: str


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_all_hrv_summaries(config: Config) -> Tuple[pd.DataFrame, List[Path]]:
    summaries_dir = config.get_hrv_summaries_dir()
    paths = sorted(summaries_dir.glob("*_hrv_summary.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No HRV summary CSV files found under: {summaries_dir}\n"
            "Run Step 2 first (run_hrv_metrics.py) to generate summaries."
        )

    dfs: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_file"] = p.name
        if "session_id" not in df.columns or df["session_id"].isna().all():
            # Fallback: infer from filename
            session_id = p.name.replace("_hrv_summary.csv", "")
            df["session_id"] = session_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, sort=False), paths


def ensure_conditions_csv(
    config: Config,
    session_ids: Sequence[str],
    conditions_path: Path,
    verbose: bool,
) -> bool:
    """Ensure conditions.csv exists.

    Returns:
        True if conditions.csv exists and should be used.
        False if a template was created and the program should exit.
    """

    if conditions_path.exists():
        return True

    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        {
            "session_id": sorted(set(session_ids)),
            "group": "",
            "Set1_Cond": "",
            "Set2_Cond": "",
        }
    )
    template.to_csv(conditions_path, index=False)

    if verbose:
        print(f"Data/conditions.csv を作成しました: {conditions_path}")
        print("グループと条件を記入してから再実行してください。")
        print("必須列: session_id, group, Set1_Cond, Set2_Cond")
        print("group は Increase / Decrease")
        print("Set*_Cond は Increase / Decrease / Control")

    return False


def load_conditions(conditions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(conditions_path)
    required = {"session_id", "group", "Set1_Cond", "Set2_Cond"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"conditions.csv is missing required columns: {sorted(missing)}\n"
            f"Path: {conditions_path}"
        )

    df = df.copy()
    df["session_id"] = df["session_id"].astype(str)
    return df


_SEGMENT_RE = re.compile(r"^(?P<idx>\d+)[_](?P<core>.+)$")


def parse_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "segment_name" not in df.columns:
        raise ValueError("Input HRV summary CSVs must contain 'segment_name' column")

    segment_name = df["segment_name"].astype(str)
    match = segment_name.str.extract(_SEGMENT_RE)

    df["segment_idx"] = pd.to_numeric(match["idx"], errors="coerce")
    df["segment_core"] = match["core"].fillna(segment_name)

    # Exclusion: Practice
    df["exclude_practice"] = df["segment_core"].str.contains("Practice", case=False, na=False)

    # Phase mapping
    core = df["segment_core"].astype(str)
    df["Phase"] = np.select(
        [
            core.str.contains("Resting", na=False),
            core.str.contains("GoNoGo", na=False),
            core.str.contains("Session", na=False),
        ],
        [
            "Resting",
            "GoNoGo_Task",
            "HR_Feedback",
        ],
        default="Unknown",
    )

    return df


def _extract_set_id_from_core(core: str) -> Optional[int]:
    if "Set1" in core:
        return 1
    if "Set2" in core:
        return 2
    return None


def _explicit_session_condition(core: str) -> Optional[str]:
    # Priority: explicit condition names
    if "Session_Control" in core:
        return "Control"
    if "Session_Increase" in core:
        return "Increase"
    if "Session_Decrease" in core:
        return "Decrease"
    return None


def add_protocol_mappings(df: pd.DataFrame, conditions: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    df = df.copy()

    # Drop practice first
    n_before = len(df)
    df = df[~df["exclude_practice"]].copy()
    if verbose:
        removed = n_before - len(df)
        if removed:
            print(f"Excluded Practice segments: {removed}")

    df = df.merge(conditions, on="session_id", how="left", validate="many_to_one")

    # Track sessions without metadata
    missing_meta = df["group"].isna() | (df["group"].astype(str).str.strip() == "")
    if missing_meta.any() and verbose:
        missing_sessions = sorted(df.loc[missing_meta, "session_id"].dropna().unique().tolist())
        print(f"Warning: conditions.csv has missing entries for sessions: {missing_sessions}")

    # Determine target label from group
    df["target_label"] = np.where(df["group"].astype(str) == "Increase", "Increase", "Decrease")

    # Set id for segments that contain explicit Set tokens
    df["set_id"] = df["segment_core"].astype(str).apply(_extract_set_id_from_core)

    # Base condition_raw
    df["condition_raw"] = None

    core = df["segment_core"].astype(str)

    # Resting mapping
    is_rest = df["Phase"] == "Resting"
    is_rest_hr1 = is_rest & core.str.contains("Resting_HR_1", na=False)
    is_rest_hr2 = is_rest & core.str.contains("Resting_HR_2", na=False)
    df.loc[is_rest_hr1, "condition_raw"] = "Baseline"

    # GoNoGo mapping
    is_gng = df["Phase"] == "GoNoGo_Task"
    is_gng_base = is_gng & core.str.contains("GoNoGo_Baseline", na=False)
    is_gng_set = is_gng & core.str.contains("GoNoGo_Set", na=False)
    df.loc[is_gng_base, "condition_raw"] = "Baseline"

    # HR_Feedback mapping
    is_sess = df["Phase"] == "HR_Feedback"
    df.loc[is_sess, "condition_raw"] = df.loc[is_sess, "segment_core"].astype(str).apply(_explicit_session_condition)

    # Apply Set mapping where needed
    def _cond_from_set_row(row: pd.Series) -> Optional[str]:
        set_id = row.get("set_id")
        if pd.isna(set_id):
            return None
        if int(set_id) == 1:
            return row.get("Set1_Cond")
        if int(set_id) == 2:
            return row.get("Set2_Cond")
        return None

    needs_set_map = (
        (is_rest_hr2 | is_gng_set)
        | (is_sess & df["condition_raw"].isna())
    )

    df.loc[needs_set_map, "condition_raw"] = df.loc[needs_set_map].apply(_cond_from_set_row, axis=1)

    # For Session_01/02: set_id might be missing if the row is Session_01/02 (no Set token)
    # Infer set_id for Session_01/02
    def _infer_session_set_id(core_value: str) -> Optional[int]:
        if "Session_01" in core_value:
            return 1
        if "Session_02" in core_value:
            return 2
        return None

    is_sess_numbered = is_sess & core.str.contains(r"Session_0[12]", regex=True, na=False)
    df.loc[is_sess_numbered, "set_id"] = df.loc[is_sess_numbered, "segment_core"].astype(str).apply(_infer_session_set_id)

    # Handle mixed format: propagate set_id from numbered Session_01/02 to same segment_idx rows
    # so that e.g., 04_Session_Control can inherit set_id from 04_Session_01
    def _fill_set_id_within_index(g: pd.DataFrame) -> pd.DataFrame:
        if g["set_id"].notna().any():
            inferred = g["set_id"].dropna().iloc[0]
            g.loc[g["set_id"].isna(), "set_id"] = inferred
        return g

    sess_rows = df[is_sess].copy()
    if "segment_idx" in sess_rows.columns:
        filled = (
            sess_rows.groupby(["session_id", "segment_idx"], dropna=False, group_keys=False)
            .apply(_fill_set_id_within_index)
        )
        df.loc[filled.index, "set_id"] = filled["set_id"]

    # Exclusion rule for Session: neither explicit condition nor numbered session (after fill)
    is_sess_unmappable = is_sess & df["condition_raw"].isna()
    df["exclude_unmappable_session"] = is_sess_unmappable

    if verbose and is_sess_unmappable.any():
        examples = (
            df.loc[is_sess_unmappable, ["session_id", "segment_name"]]
            .drop_duplicates()
            .head(10)
        )
        print("Warning: Unmappable Session segments will be excluded (no explicit condition, no Session_01/02):")
        for _, r in examples.iterrows():
            print(f"  - {r['session_id']}: {r['segment_name']}")

    df = df[~df["exclude_unmappable_session"]].copy()

    # Map condition_raw -> condition (Baseline / Control / Target)
    df["condition_raw"] = df["condition_raw"].astype(str)

    def _to_condition(row: pd.Series) -> Optional[str]:
        raw = str(row.get("condition_raw", "")).strip()
        if raw in {"", "nan", "None"}:
            return None
        if raw == "Baseline":
            return "Baseline"
        if raw == "Control":
            return "Control"

        target_label = str(row.get("target_label", "")).strip()
        if raw == target_label:
            return "Target"

        # Non-target intervention label for this group
        return None

    df["Condition"] = df.apply(_to_condition, axis=1)

    # Drop rows where Condition could not be determined (e.g., Increase label in Decrease group)
    unmapped = df["Condition"].isna()
    if unmapped.any() and verbose:
        ex = df.loc[unmapped, ["session_id", "segment_name", "group", "condition_raw"]].drop_duplicates().head(10)
        print("Warning: Some rows could not be mapped to Baseline/Control/Target and will be excluded:")
        for _, r in ex.iterrows():
            print(f"  - {r['session_id']}: {r['segment_name']} (group={r['group']}, raw={r['condition_raw']})")

    df = df[~unmapped].copy()

    return df


def _paired_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float | str]:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    n = int(len(x))
    if n < 3:
        return {
            "n": n,
            "normality_p": float("nan"),
            "test": "N/A",
            "stat": float("nan"),
            "p": float("nan"),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "sd_diff": float("nan"),
            "cohens_dz": float("nan"),
        }

    diff = y - x
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    cohens_dz = mean_diff / sd_diff if sd_diff and np.isfinite(sd_diff) and sd_diff > 0 else float("nan")

    # Normality check on paired differences
    try:
        normality_p = float(stats.shapiro(diff).pvalue)
    except Exception:
        normality_p = float("nan")

    if np.isfinite(normality_p) and normality_p >= alpha:
        test = "paired_t"
        res = stats.ttest_rel(y, x, nan_policy="omit")
        stat = float(res.statistic)
        p = float(res.pvalue)
    else:
        test = "wilcoxon"
        try:
            res = stats.wilcoxon(y, x, alternative="two-sided")
            stat = float(res.statistic)
            p = float(res.pvalue)
        except ValueError:
            # Likely all differences are zero
            stat = 0.0
            p = 1.0

    return {
        "n": n,
        "normality_p": normality_p,
        "test": test,
        "stat": stat,
        "p": p,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "sd_diff": sd_diff,
        "cohens_dz": cohens_dz,
    }


def _build_wide_values(
    df: pd.DataFrame,
    phase: str,
    group: str,
    metric: str,
    verbose: bool,
) -> pd.DataFrame:
    """Build a wide table per phase with columns needed for plotting and pairing.

    Returns a DataFrame indexed by session_id with columns among:
        - Baseline, Control, Target
    depending on phase.
    """

    d = df[(df["Phase"] == phase) & (df["group"] == group)].copy()
    if d.empty:
        return pd.DataFrame(columns=["session_id", "Baseline", "Control", "Target"]).set_index("session_id")

    if metric not in d.columns:
        raise ValueError(f"Metric column not found: {metric}")

    # Keep only needed columns
    d = d[["session_id", "segment_core", "set_id", "Condition", metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")

    if phase == "Resting":
        # Baseline: Resting_HR_1_Set1/2 (set-specific)
        is_base = d["segment_core"].str.contains("Resting_HR_1", na=False)
        is_post = d["segment_core"].str.contains("Resting_HR_2", na=False)

        base = d[is_base & d["set_id"].notna()].copy()
        post = d[is_post & d["set_id"].notna() & d["Condition"].isin(["Control", "Target"])].copy()

        # Determine per-session control/target set_id from post segments
        ctrl_set = (
            post[post["Condition"] == "Control"]
            .dropna(subset=["set_id"])
            .groupby("session_id")["set_id"]
            .first()
        )
        targ_set = (
            post[post["Condition"] == "Target"]
            .dropna(subset=["set_id"])
            .groupby("session_id")["set_id"]
            .first()
        )

        # Post values per condition
        ctrl_val = (
            post[post["Condition"] == "Control"]
            .groupby("session_id")[metric]
            .first()
        )
        targ_val = (
            post[post["Condition"] == "Target"]
            .groupby("session_id")[metric]
            .first()
        )

        # Baseline mean for plotting (mean across available Resting_HR_1_Set1/2)
        baseline_mean = base.groupby("session_id")[metric].mean()

        out = pd.DataFrame({"Baseline": baseline_mean, "Control": ctrl_val, "Target": targ_val})
        out.index.name = "session_id"
        return out

    if phase == "GoNoGo_Task":
        is_base = d["segment_core"].str.contains("GoNoGo_Baseline", na=False)
        is_set = d["segment_core"].str.contains("GoNoGo_Set", na=False)

        base = d[is_base].copy()
        post = d[is_set & d["Condition"].isin(["Control", "Target"])].copy()

        baseline_val = base.groupby("session_id")[metric].first()
        ctrl_val = post[post["Condition"] == "Control"].groupby("session_id")[metric].first()
        targ_val = post[post["Condition"] == "Target"].groupby("session_id")[metric].first()

        out = pd.DataFrame({"Baseline": baseline_val, "Control": ctrl_val, "Target": targ_val})
        out.index.name = "session_id"
        return out

    if phase == "HR_Feedback":
        # No baseline comparisons
        post = d[d["Condition"].isin(["Control", "Target"])].copy()
        ctrl_val = post[post["Condition"] == "Control"].groupby("session_id")[metric].first()
        targ_val = post[post["Condition"] == "Target"].groupby("session_id")[metric].first()
        out = pd.DataFrame({"Control": ctrl_val, "Target": targ_val})
        out.index.name = "session_id"
        return out

    return pd.DataFrame(columns=["session_id"]).set_index("session_id")


def _run_all_stats(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    groups = ["Increase", "Decrease"]
    phases = ["Resting", "GoNoGo_Task", "HR_Feedback"]

    for group in groups:
        for phase in phases:
            for metric in METRICS:
                wide = _build_wide_values(df, phase=phase, group=group, metric=metric, verbose=verbose)

                if phase in {"Resting", "GoNoGo_Task"}:
                    comps = [
                        ("Baseline", "Control"),
                        ("Baseline", "Target"),
                        ("Control", "Target"),
                    ]
                else:
                    comps = [("Control", "Target")]

                for a, b in comps:
                    if a not in wide.columns or b not in wide.columns:
                        continue

                    x = wide[a].to_numpy(dtype=float)
                    y = wide[b].to_numpy(dtype=float)
                    res = _paired_test(x, y)

                    rows.append(
                        {
                            "group": group,
                            "phase": phase,
                            "metric": metric,
                            "A": a,
                            "B": b,
                            **res,
                        }
                    )

    return pd.DataFrame(rows)


def _plot_slope(
    wide: pd.DataFrame,
    phase: str,
    group: str,
    metric: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Determine order based on phase
    if phase in {"Resting", "GoNoGo_Task"}:
        order = ["Baseline", "Control", "Target"]
    else:
        order = ["Control", "Target"]

    plot_df = wide[order].copy()
    plot_df = plot_df.dropna(how="any")

    if plot_df.empty:
        return

    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for _, row in plot_df.iterrows():
        y = [row[c] for c in order]
        ax.plot(x, y, marker="o", linewidth=1.0, alpha=0.5)

    # Overlay mean points
    means = [float(np.nanmean(plot_df[c].to_numpy(dtype=float))) for c in order]
    ax.plot(x, means, color="black", marker="o", linewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title(f"{group} | {phase} | {metric}")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_box(
    wide: pd.DataFrame,
    phase: str,
    group: str,
    metric: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if phase in {"Resting", "GoNoGo_Task"}:
        order = ["Baseline", "Control", "Target"]
    else:
        order = ["Control", "Target"]

    plot_df = wide[order].copy().reset_index(names=["session_id"])
    plot_long = plot_df.melt(id_vars=["session_id"], value_vars=order, var_name="Condition", value_name="value")
    plot_long = plot_long.dropna(subset=["value"]).copy()

    if plot_long.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    sns.boxplot(data=plot_long, x="Condition", y="value", order=order, ax=ax, showfliers=False)
    sns.stripplot(
        data=plot_long,
        x="Condition",
        y="value",
        order=order,
        ax=ax,
        color="black",
        alpha=0.6,
        jitter=0.2,
        size=3,
    )

    ax.set_title(f"{group} | {phase} | {metric}")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _run_all_plots(df: pd.DataFrame, config: Config, verbose: bool) -> None:
    out_dir = config.get_stats_dir()

    groups = ["Increase", "Decrease"]
    phases = ["Resting", "GoNoGo_Task", "HR_Feedback"]

    for group in groups:
        for phase in phases:
            for metric in METRICS:
                wide = _build_wide_values(df, phase=phase, group=group, metric=metric, verbose=verbose)
                if wide.empty:
                    continue

                slope_path = out_dir / f"slope_{group}_{phase}_{metric}.png"
                box_path = out_dir / f"box_{group}_{phase}_{metric}.png"

                _plot_slope(wide, phase=phase, group=group, metric=metric, out_path=slope_path)
                _plot_box(wide, phase=phase, group=group, metric=metric, out_path=box_path)

                if verbose:
                    print(f"Saved: {slope_path.name}")
                    print(f"Saved: {box_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3: Group Statistical Analysis & Visualization (Protocol v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run group stats for all sessions in Results/hrv_summaries
    python src/run_group_stats.py

    # Quiet mode
    python src/run_group_stats.py --quiet
        """,
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    config = Config()

    # Discover sessions from filenames first (required for template generation)
    summaries_dir = config.get_hrv_summaries_dir()
    summary_paths = sorted(summaries_dir.glob("*_hrv_summary.csv"))
    if not summary_paths:
        print(f"Error: No HRV summary CSVs found under {summaries_dir}")
        print("Run Step 2 first: python src/run_hrv_metrics.py --session <SESSION>")
        return 1

    session_ids = [p.name.replace("_hrv_summary.csv", "") for p in summary_paths]

    conditions_path = config.get_data_dir() / "conditions.csv"
    if not ensure_conditions_csv(config, session_ids, conditions_path, verbose=verbose):
        return 0

    if verbose:
        print("=" * 60)
        print("Group Statistical Analysis - Step 3")
        print("=" * 60)
        print(f"HRV summaries: {summaries_dir}")
        print(f"conditions.csv : {conditions_path}")

    # Load data
    df_all, _ = load_all_hrv_summaries(config)
    if verbose:
        print(f"Loaded HRV rows: {len(df_all)}")

    df_all = parse_segments(df_all)

    conditions = load_conditions(conditions_path)
    df_mapped = add_protocol_mappings(df_all, conditions, verbose=verbose)

    # Run stats
    results = _run_all_stats(df_mapped, verbose=verbose)

    out_dir = config.get_stats_dir()
    results_path = out_dir / "group_stats_results.csv"
    results.to_csv(results_path, index=False)

    if verbose:
        print(f"Saved stats: {results_path}")

    # Plots
    _run_all_plots(df_mapped, config=config, verbose=verbose)

    if verbose:
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
