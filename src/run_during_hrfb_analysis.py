#!/usr/bin/env python3
"""End-to-end analysis for *during* HR feedback (HRFB).

This script mirrors the structure of `run_resting_hr_feedback_analysis.py`, but
uses Timestamp_2.csv columns HRFB_1 / HRFB_2 to extract during-feedback windows.

For each HRFB start time (HRFB_1 and HRFB_2), we compute two 5-min windows from
a 15-min feedback period:
- First5:  start + 0 min  to + 5 min
- Last5:   start + 10 min to + 15 min

Each window is extracted with ±padding for filter edge artifacts, then processed
via:
- `run_signal_diagnosis.process_segment` (bandpass + trimming + R-peak detection)
- `run_hrv_metrics.process_peaks_file` (HRV + mean HR)

Outputs per session:
- Results/<session_id>/during_hrfb_metrics.csv

Notes
-----
- Missing/failed segments are kept as rows with NaN metrics and status/error_msg
  so downstream analysis can distinguish missingness vs failures.
- Segment file names use a physical identifier (e.g., during_HRFB1_First5) and
  do not encode Target/Control. Semantic condition labels are stored in CSV.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure scripts work when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config
from extract_ecg_columns import extract_columns
from run_signal_diagnosis import process_segment
from run_hrv_metrics import process_peaks_file
from split_by_feedback_timestamp import resolve_session_identity, split_segment_by_start_and_duration


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "Data"


def _results_dir() -> Path:
    return _project_root() / "Results"


def _raw_session_dir(session_id: str) -> Path:
    return _data_dir() / "Raw" / session_id


def _extracted_path(session_id: str) -> Path:
    return _data_dir() / "Processed" / session_id / "extracted" / f"{session_id}_ext.csv"


def _clean_hhmm_cell(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "nat"}:
        return None
    text = text.replace('"', "").replace("'", "").strip()
    if not text.isdigit():
        return None
    return int(text)


def ensure_extracted_csv(*, session_id: str, rebuild: bool) -> Path:
    raw_dir = _raw_session_dir(session_id)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    output_path = _extracted_path(session_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not rebuild:
        return output_path

    if output_path.exists():
        output_path.unlink()

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {raw_dir}")

    for txt in txt_files:
        extract_columns(
            input_path=txt,
            output_path=output_path,
            chunk_size=200_000,
            delimiter="\t",
            skiprows=4,
            thousands=",",
        )

    return output_path


def _read_subject_order(*, date_str: str, subject_id: str, data_dir: Path) -> Optional[int]:
    subject_df = pd.read_csv(data_dir / "Subject" / "SubjectData.csv")
    row_df = subject_df[(subject_df["Date"] == date_str) & (subject_df["Subject"] == subject_id)]
    if row_df.empty:
        return None
    try:
        return int(row_df.iloc[0]["Order"])
    except Exception:
        return None


def _read_hrfb_hhmm(*, date_str: str, subject_id: str, data_dir: Path) -> Dict[str, Optional[int]]:
    ts_path = data_dir / "Timestamp" / "Timestamp_2.csv"
    ts_df = pd.read_csv(ts_path)

    row_df = ts_df[(ts_df["Date"] == date_str) & (ts_df["Subject"] == subject_id)]
    if row_df.empty:
        return {"HRFB_1": None, "HRFB_2": None}

    row = row_df.iloc[0]
    return {
        "HRFB_1": _clean_hhmm_cell(row.get("HRFB_1")),
        "HRFB_2": _clean_hhmm_cell(row.get("HRFB_2")),
    }


def _hrfb_index_to_condition(*, order: Optional[int], hrfb_index: int) -> str:
    """Map HRFB_1/2 to Target/Control using Order.

    Order=1: HRFB_1=Target, HRFB_2=Control
    Order=2: HRFB_1=Control, HRFB_2=Target
    """
    if order == 1:
        return "Target" if hrfb_index == 1 else "Control"
    if order == 2:
        return "Control" if hrfb_index == 1 else "Target"
    return "Unknown"


def _metrics_to_time_mean_hr(metrics_dict: dict) -> float:
    # Preferred key
    for key in [
        "time_mean_hr",
        "mean_hr",
        "meanHR",
        "hr_mean",
    ]:
        if key in metrics_dict:
            try:
                return float(metrics_dict[key])
            except Exception:
                pass
    return float("nan")


def run_session(*, session_id: str, rebuild_extracted: bool, quiet: bool) -> Path:
    config = Config()
    data_dir = _data_dir()

    if not quiet:
        print("=" * 70)
        print(f"During HRFB Analysis: {session_id}")
        print("=" * 70)

    extracted_csv = ensure_extracted_csv(session_id=session_id, rebuild=rebuild_extracted)

    # Resolve identity using existing logic (handles MS vs MS_1 etc.)
    session = resolve_session_identity(session_id=session_id, data_dir=data_dir)
    order = _read_subject_order(date_str=session.date_str, subject_id=session.subject_id, data_dir=data_dir)

    hrfb = _read_hrfb_hhmm(date_str=session.date_str, subject_id=session.subject_id, data_dir=data_dir)

    # Segment definitions
    segment_specs: List[dict] = []
    for hrfb_index in [1, 2]:
        col = f"HRFB_{hrfb_index}"
        hhmm = hrfb.get(col)
        for phase, start_offset_sec in [("First5", 0.0), ("Last5", 10.0 * 60.0)]:
            segment_specs.append(
                {
                    "hrfb_index": hrfb_index,
                    "timestamp_col": col,
                    "start_hhmm": hhmm,
                    "phase": phase,
                    "start_offset_sec": start_offset_sec,
                    "duration_sec": 5.0 * 60.0,
                }
            )

    rows: List[dict] = []

    for spec in segment_specs:
        hrfb_index = int(spec["hrfb_index"])
        phase = str(spec["phase"])  # First5 / Last5
        start_hhmm = spec["start_hhmm"]
        start_offset_sec = float(spec["start_offset_sec"])
        duration_sec = float(spec["duration_sec"])

        segment_name = f"during_HRFB{hrfb_index}_{phase}"
        condition = _hrfb_index_to_condition(order=order, hrfb_index=hrfb_index)

        base_row = {
            "session_id": session_id,
            "Date": session.date_str,
            "Subject": session.subject_id,
            "BF_Type": session.bf_type,
            "Order": order,
            "hrfb_index": hrfb_index,
            "timestamp_col": spec["timestamp_col"],
            "phase": phase,
            "condition": condition,
            "segment_key": segment_name,
            "time_mean_hr": np.nan,
            "status": "pending",
            "error_msg": "",
            "quality_notes": "",
        }

        # Missing timestamp row / value
        if start_hhmm is None:
            row = base_row.copy()
            row["status"] = "missing_timestamp"
            row["error_msg"] = f"{spec['timestamp_col']} is missing/unparseable in Timestamp_2.csv"
            rows.append(row)
            continue

        # 1) Split
        try:
            segment_csv = split_segment_by_start_and_duration(
                session_id=session_id,
                extracted_csv_path=extracted_csv,
                segment_key=segment_name,
                start_hhmm=start_hhmm,
                start_offset_sec=start_offset_sec,
                duration_sec=duration_sec,
                buffer_sec=config.FILTER_PADDING_SEC,
                allow_missing=True,
            )
        except Exception as e:
            row = base_row.copy()
            row["status"] = "split_failed"
            row["error_msg"] = f"split_failed: {e}"
            rows.append(row)
            if not quiet:
                print(f"  ✗ Split failed: {segment_name}: {e}")
            continue

        if segment_csv is None:
            row = base_row.copy()
            row["status"] = "missing_segment"
            row["error_msg"] = f"No rows extracted for {segment_name}"
            rows.append(row)
            if not quiet:
                print(f"  ⚠ Missing segment rows: {segment_name}")
            continue

        # 2) Diagnosis + peaks
        try:
            ok = process_segment(segment_csv, session_id, config=config, verbose=(not quiet))
            if not ok:
                raise RuntimeError("process_segment returned False")
        except Exception as e:
            row = base_row.copy()
            row["status"] = "diagnosis_failed"
            row["error_msg"] = f"diagnosis_failed: {e}"
            rows.append(row)
            if not quiet:
                print(f"  ✗ Diagnosis failed: {segment_name}: {e}")
                traceback.print_exc()
            continue

        # 3) Metrics
        try:
            peaks_path = config.get_peaks_path(session_id, segment_csv.stem)
            metrics = process_peaks_file(peaks_path, config=config, verbose=(not quiet))
            flat = metrics.to_flat_dict()

            row = base_row.copy()
            row.update(flat)
            row["time_mean_hr"] = _metrics_to_time_mean_hr(flat)
            row["quality_notes"] = str(flat.get("quality_notes", ""))
            row["status"] = "ok"
            rows.append(row)
        except Exception as e:
            row = base_row.copy()
            row["status"] = "metrics_failed"
            row["error_msg"] = f"metrics_failed: {e}"
            rows.append(row)
            if not quiet:
                print(f"  ✗ Metrics failed: {segment_name}: {e}")
                traceback.print_exc()

    out_df = pd.DataFrame(rows)

    # Make sure required output columns exist (even if metrics were missing)
    required_cols = [
        "session_id",
        "Date",
        "Subject",
        "BF_Type",
        "Order",
        "hrfb_index",
        "condition",
        "phase",
        "time_mean_hr",
        "status",
        "error_msg",
        "quality_notes",
        "segment_key",
    ]
    for c in required_cols:
        if c not in out_df.columns:
            out_df[c] = np.nan

    out_path = _results_dir() / session_id / "during_hrfb_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if not quiet:
        ok_n = int((out_df["status"] == "ok").sum()) if "status" in out_df.columns else 0
        print(f"Saved: {out_path} (ok={ok_n}/{len(out_df)})")

    return out_path


def _iter_sessions_from_raw(data_dir: Path) -> List[str]:
    raw_dir = data_dir / "Raw"
    if not raw_dir.exists():
        return []
    return sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description="During HRFB analysis (First5 vs Last5).")
    parser.add_argument("--session", type=str, default=None, help="Session ID like 251216_TK (default: all sessions)")
    parser.add_argument("--rebuild-extracted", action="store_true", help="Rebuild extracted *_ext.csv")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = parser.parse_args()

    data_dir = _data_dir()

    session_ids = [args.session] if args.session else _iter_sessions_from_raw(data_dir)
    if not session_ids:
        raise RuntimeError(f"No sessions found under: {data_dir / 'Raw'}")

    failures: List[str] = []
    for sid in session_ids:
        try:
            run_session(session_id=sid, rebuild_extracted=args.rebuild_extracted, quiet=args.quiet)
        except Exception as e:
            failures.append(sid)
            if not args.quiet:
                print(f"✗ ERROR session={sid}: {e}")
                traceback.print_exc()

    if failures and not args.quiet:
        print("\nFailed sessions:")
        for sid in failures:
            print(f"  - {sid}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
