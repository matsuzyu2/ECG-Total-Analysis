#!/usr/bin/env python3
"""Report Good Responders for Biofeedback (BF) based on Difference-in-Differences.

This script scans per-session metrics CSV files and identifies subjects whose
Target condition outperformed Control, according to BF_Type:

- Control_Delta = Control_Post - Control_Pre
- Target_Delta  = Target_Post  - Target_Pre
- Diff          = Target_Delta - Control_Delta

Good responder criteria:
- BF_Type == 'Dec': Diff <= -threshold
- BF_Type == 'Inc': Diff >=  threshold

Inputs (auto-detected):
- Preferred: Results/<session>/resting_hr_feedback_metrics.csv (created by run_resting_hr_feedback_analysis.py)
- Fallbacks:
  - Data/Processed/<session>/metrics.csv
  - Data/Processed/<session>/resting_hr_feedback_metrics.csv

Also merges subject metadata from:
- Data/Subject/SubjectData.csv

Usage:
  python src/report_good_responders.py --threshold 0.0
  python src/report_good_responders.py --threshold 1.0 --output Results/good_responders.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config


@dataclass(frozen=True)
class DeltasRow:
    subject_id: str
    session_id: str
    bf_type: str
    control_delta: float
    target_delta: float
    diff: float


def _safe_float(x: object) -> float:
    try:
        # Cast through str to satisfy static typing while staying robust.
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _get_mean_hr(metrics_df: pd.DataFrame, *, condition: str, phase: str) -> float:
    row = metrics_df[(metrics_df["condition"] == condition) & (metrics_df["phase"] == phase)]
    if row.empty:
        return float("nan")
    return _safe_float(row.iloc[0].get("time_mean_hr"))


def _read_deltas_from_metrics_csv(metrics_csv: Path) -> Optional[DeltasRow]:
    df = pd.read_csv(metrics_csv)

    required = {"condition", "phase", "time_mean_hr", "BF_Type", "Subject"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    # BF_Type / Subject are duplicated per segment; take the first non-null.
    bf_type = str(df["BF_Type"].dropna().iloc[0]) if df["BF_Type"].notna().any() else ""
    subject_id = str(df["Subject"].dropna().iloc[0]) if df["Subject"].notna().any() else ""
    session_id = metrics_csv.parent.name

    if not subject_id:
        return None

    control_pre = _get_mean_hr(df, condition="control", phase="pre")
    control_post = _get_mean_hr(df, condition="control", phase="post")
    target_pre = _get_mean_hr(df, condition="target", phase="pre")
    target_post = _get_mean_hr(df, condition="target", phase="post")

    control_delta = control_post - control_pre if pd.notna(control_pre) and pd.notna(control_post) else float("nan")
    target_delta = target_post - target_pre if pd.notna(target_pre) and pd.notna(target_post) else float("nan")
    diff = target_delta - control_delta if pd.notna(target_delta) and pd.notna(control_delta) else float("nan")

    return DeltasRow(
        subject_id=subject_id,
        session_id=session_id,
        bf_type=bf_type,
        control_delta=float(control_delta),
        target_delta=float(target_delta),
        diff=float(diff),
    )


def _iter_metrics_csvs(config: Config) -> Iterable[Path]:
    seen: set[Path] = set()
    # Preferred location: Results/<session>/resting_hr_feedback_metrics.csv
    results_dir = config.get_results_dir()
    if results_dir.exists():
        for session_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
            p = session_dir / "resting_hr_feedback_metrics.csv"
            if p.exists() and p not in seen:
                seen.add(p)
                yield p

    # Fallbacks under Data/Processed/<session>/
    processed_dir = config.get_data_dir() / config.PROCESSED_SUBDIR
    if processed_dir.exists():
        for session_dir in sorted([p for p in processed_dir.iterdir() if p.is_dir()]):
            for name in ("metrics.csv", "resting_hr_feedback_metrics.csv"):
                p = session_dir / name
                if p.exists() and p not in seen:
                    seen.add(p)
                    yield p


def _sort_good_responders(df: pd.DataFrame) -> pd.DataFrame:
    """Sort good responders in a 'better first' order.

    - Dec: more negative Diff is better -> ascending diff
    - Inc: more positive Diff is better -> descending diff
    """

    out = df.copy()
    out["_rank"] = out["diff"]
    out.loc[out["BF_Type"] == "Inc", "_rank"] = -out.loc[out["BF_Type"] == "Inc", "diff"]
    out = out.sort_values(["BF_Type", "_rank", "Subject"], ascending=[True, True, True])
    return out.drop(columns=["_rank"], errors="ignore")


def _load_subject_metadata(config: Config) -> Optional[pd.DataFrame]:
    subject_csv = config.get_data_dir() / "Subject" / "SubjectData.csv"
    if not subject_csv.exists():
        return None

    meta = pd.read_csv(subject_csv)
    if "Subject" not in meta.columns:
        return None

    meta = meta.copy()
    meta["Subject"] = meta["Subject"].astype(str)
    return meta


def _is_good_responder(*, bf_type: str, diff: float, threshold: float) -> bool:
    if pd.isna(diff):
        return False
    if bf_type == "Dec":
        return diff <= -threshold
    if bf_type == "Inc":
        return diff >= threshold
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Good Responders based on resting HR feedback metrics (Difference-in-Differences). "
            "Scans Results/<session>/resting_hr_feedback_metrics.csv across sessions."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for responder decision (default: 0.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path. If omitted, prints a readable table to stdout.",
    )

    args = parser.parse_args(argv)
    threshold = abs(float(args.threshold))

    config = Config()

    rows: list[dict] = []
    skipped = 0
    metrics_paths = list(_iter_metrics_csvs(config))
    if not metrics_paths:
        print(
            "No metrics CSV files found. Expected e.g. Results/<session>/resting_hr_feedback_metrics.csv",
            file=sys.stderr,
        )
        return 2

    for metrics_csv in metrics_paths:
        try:
            d = _read_deltas_from_metrics_csv(metrics_csv)
        except Exception as e:
            print(f"⚠ Skipping {metrics_csv}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if d is None:
            continue

        rows.append(
            {
                "Subject": d.subject_id,
                "session_id": d.session_id,
                "BF_Type": d.bf_type,
                "control_delta": d.control_delta,
                "target_delta": d.target_delta,
                "diff": d.diff,
                "threshold": threshold,
                "is_good_responder": _is_good_responder(bf_type=d.bf_type, diff=d.diff, threshold=threshold),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid rows computed from metrics CSVs.", file=sys.stderr)
        return 2

    # Keep only Inc/Dec (unknown BF_Type should not be judged)
    df = df[df["BF_Type"].isin(["Inc", "Dec"])].copy()

    # Merge subject metadata (optional)
    meta = _load_subject_metadata(config)
    if meta is not None:
        df = df.merge(meta, how="left", on="Subject", suffixes=("", "_meta"))

    # Extract good responders
    good = df[df["is_good_responder"]].copy()
    good = _sort_good_responders(good)

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # User-facing CSV: keep only core columns up to `diff`.
        keep_cols = ["Subject", "session_id", "BF_Type", "control_delta", "target_delta", "diff"]
        good_out = good.loc[:, [c for c in keep_cols if c in good.columns]].copy()
        good_out.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} (n={len(good)}, scanned={len(metrics_paths)}, skipped={skipped})")
        return 0

    # Pretty stdout (compact; for full columns use --output)
    display_cols = ["Subject", "session_id", "BF_Type", "control_delta", "target_delta", "diff"]
    rename_for_stdout: dict[str, str] = {}
    # Keep stdout compact; full metadata is available via --output.
    for c in ["Age", "Sex"]:
        if c in good.columns:
            display_cols.append(c)

    if good.empty:
        print(f"Good responders: 0 (threshold={threshold}, scanned={len(metrics_paths)}, skipped={skipped})")
        return 0

    # Numeric formatting
    fmt = good.copy()
    for c in ["control_delta", "target_delta", "diff"]:
        if c in fmt.columns:
            fmt[c] = pd.to_numeric(fmt[c], errors="coerce")

    with pd.option_context(
        "display.max_rows",
        10_000,
        "display.max_columns",
        200,
        "display.width",
        220,
        "display.expand_frame_repr",
        False,
    ):
        print(f"Good responders (n={len(good)}, threshold={threshold}, scanned={len(metrics_paths)}, skipped={skipped}):")
        out_view = fmt[display_cols].rename(columns=rename_for_stdout)
        print(out_view.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if meta is not None:
            print("\nNote: stdout is compact. Use --output to write full merged columns.")

    # Small summary by BF_Type
    try:
        summary = good.groupby("BF_Type").size().rename("n").reset_index()
        print("\nBy BF_Type:")
        print(summary.to_string(index=False))
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
