#!/usr/bin/env python3
"""Create a consolidated CSV listing sessions with missing Feedback segments.

This script scans existing per-session outputs under Results/{session}/
(resting_hr_feedback_metrics.csv) and reports which of the 4 required segments
are missing.

Output (default): Results/resting_hr_feedback_missing_segments.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_KEYS = [
    "Feedback_Con_pre",
    "Feedback_Con_post",
    "Feedback_pre",
    "Feedback_post",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_root() -> Path:
    return _project_root() / "Results"


def build_missing_report(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    if not results_root.exists():
        return pd.DataFrame(rows)

    for session_dir in sorted([p for p in results_root.iterdir() if p.is_dir() and p.name[:3].isdigit()]):
        metrics_path = session_dir / "resting_hr_feedback_metrics.csv"
        if not metrics_path.exists():
            # No metrics => treat as missing all segments (best-effort)
            for key in REQUIRED_KEYS:
                rows.append(
                    {
                        "session_id": session_dir.name,
                        "Date": None,
                        "Subject": None,
                        "BF_Type": None,
                        "missing_segment_key": key,
                        "reason": "metrics_csv_missing",
                    }
                )
            continue

        df = pd.read_csv(metrics_path)
        present = set(df.get("segment_key", pd.Series([], dtype=str)).astype(str).tolist())

        # Pull metadata from the file if available
        meta = {
            "session_id": session_dir.name,
            "Date": df["Date"].iloc[0] if "Date" in df.columns and len(df) else None,
            "Subject": df["Subject"].iloc[0] if "Subject" in df.columns and len(df) else None,
            "BF_Type": df["BF_Type"].iloc[0] if "BF_Type" in df.columns and len(df) else None,
        }

        missing = [k for k in REQUIRED_KEYS if k not in present]
        for key in missing:
            rows.append({**meta, "missing_segment_key": key, "reason": "segment_missing"})

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report missing Feedback segments across sessions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_results_root() / "resting_hr_feedback_missing_segments.csv",
        help="Output CSV path (default: Results/resting_hr_feedback_missing_segments.csv)",
    )
    args = parser.parse_args()

    df = build_missing_report(_results_root())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote: {args.output} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
