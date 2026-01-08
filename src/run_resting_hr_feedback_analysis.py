#!/usr/bin/env python3
"""End-to-end analysis for resting HR (Feedback windows).

Pipeline per session:
1) Extract required ECG columns from Data/Raw/{session}/*.txt using extract_ecg_columns.extract_columns
   - Supports multi-file sessions by appending into one extracted CSV
2) Split into 4 Feedback segments using HHMM start times from Timestamp.csv
   - Cuts 60s measurement with ±15s padding
3) Run signal diagnosis pipeline (bandpass -> trim padding -> R-peak detection)
4) Compute HRV metrics for each segment
5) Plot pre/post changes for Control and Target conditions per subject

Outputs:
- Data/Processed/{session}/extracted/{session}_ext.csv
- Data/Processed/{session}/split_segments/01_Feedback_Con_pre.csv ...
- Results/{session}/peaks_*.json and diagnosis_*.html
- Results/{session}/resting_hr_feedback_metrics.csv
- Results/{session}/resting_hr_feedback_comparison.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config
from extract_ecg_columns import extract_columns
from run_signal_diagnosis import process_segment
from run_hrv_metrics import process_peaks_file
from split_by_feedback_timestamp import split_feedback_segments, FEEDBACK_COLUMNS_ORDER


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _project_root() / "Data"


def _raw_session_dir(session_id: str) -> Path:
    return _data_dir() / "Raw" / session_id


def _extracted_path(session_id: str) -> Path:
    return _data_dir() / "Processed" / session_id / "extracted" / f"{session_id}_ext.csv"


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


def _segment_key_to_condition_phase(key: str) -> Tuple[str, str]:
    # key in {Feedback_Con_pre, Feedback_Con_post, Feedback_pre, Feedback_post}
    condition = "control" if "_Con_" in key else "target"
    phase = "pre" if key.endswith("_pre") else "post"
    return condition, phase


def _plot_pre_post(metrics_df: pd.DataFrame, *, session_id: str, bf_type: str, output_html: Path) -> None:
    # Expect metrics_df contains rows for 4 segments with time_mean_hr.
    def get_value(condition: str, phase: str) -> float:
        row = metrics_df[(metrics_df["condition"] == condition) & (metrics_df["phase"] == phase)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0]["time_mean_hr"])

    control_pre = get_value("control", "pre")
    control_post = get_value("control", "post")
    target_pre = get_value("target", "pre")
    target_post = get_value("target", "post")

    x_control = ["Control", "Control"]
    x_target = ["Target", "Target"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_control,
            y=[control_pre, control_post],
            mode="markers+lines",
            name="Control (pre→post)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_target,
            y=[target_pre, target_post],
            mode="markers+lines",
            name=f"Target (pre→post, BF_Type={bf_type})",
        )
    )

    # Add simple delta annotations
    try:
        d_control = control_post - control_pre
    except Exception:
        d_control = float("nan")
    try:
        d_target = target_post - target_pre
    except Exception:
        d_target = float("nan")

    fig.update_layout(
        title=f"Resting HR pre/post changes (session={session_id}, BF_Type={bf_type})",
        xaxis_title="Condition",
        yaxis_title="Mean HR (bpm)",
        hovermode="x unified",
    )

    fig.add_annotation(
        x="Control",
        y=max(control_pre, control_post) if pd.notna(control_pre) and pd.notna(control_post) else 0,
        text=f"Δ={d_control:+.2f} bpm" if pd.notna(d_control) else "Δ=NA",
        showarrow=False,
        yshift=20,
    )
    fig.add_annotation(
        x="Target",
        y=max(target_pre, target_post) if pd.notna(target_pre) and pd.notna(target_post) else 0,
        text=f"Δ={d_target:+.2f} bpm" if pd.notna(d_target) else "Δ=NA",
        showarrow=False,
        yshift=20,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)


def run_session(*, session_id: str, rebuild_extracted: bool, quiet: bool, allow_missing_segments: bool) -> tuple[dict, list[str]]:
    config = Config()

    if not quiet:
        print("=" * 70)
        print(f"Resting HR Feedback Analysis: {session_id}")
        print("=" * 70)

    extracted_csv = ensure_extracted_csv(session_id=session_id, rebuild=rebuild_extracted)

    session_info, segment_paths, missing_keys = split_feedback_segments(
        session_id=session_id,
        extracted_csv_path=extracted_csv,
        buffer_sec=config.FILTER_PADDING_SEC,
        measure_sec=60.0,
        allow_missing=allow_missing_segments,
    )

    if missing_keys and not quiet:
        print(f"⚠ Missing segments (skipped): {', '.join(missing_keys)}")

    # Step 1: diagnosis + peak detection
    for seg_key in FEEDBACK_COLUMNS_ORDER:
        if seg_key not in segment_paths:
            continue
        seg_path = segment_paths[seg_key]
        if not quiet:
            print(f"\n--- Diagnosis for {seg_key} ---")
        ok = process_segment(seg_path, session_id=session_id, config=config, verbose=not quiet)
        if not ok:
            raise RuntimeError(f"Diagnosis failed for segment: {seg_path}")

    # Step 2: HRV metrics
    metrics_rows: List[dict] = []
    for seg_key in FEEDBACK_COLUMNS_ORDER:
        if seg_key not in segment_paths:
            continue
        seg_path = segment_paths[seg_key]
        segment_name = seg_path.stem
        peaks_path = config.get_peaks_path(session_id, segment_name)

        metrics = process_peaks_file(peaks_path, config=config, verbose=not quiet)
        row = metrics.to_flat_dict()

        condition, phase = _segment_key_to_condition_phase(seg_key)
        row.update(
            {
                "segment_key": seg_key,
                "condition": condition,
                "phase": phase,
                "BF_Type": session_info.bf_type,
                "Subject": session_info.subject_id,
                "Date": session_info.date_str,
            }
        )
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    # Save session-local summary
    results_dir = config.get_session_results_dir(session_id)
    metrics_csv = results_dir / "resting_hr_feedback_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Plot mean HR (bpm) pre/post changes
    plot_html = results_dir / "resting_hr_feedback_comparison.html"
    _plot_pre_post(metrics_df, session_id=session_id, bf_type=session_info.bf_type, output_html=plot_html)

    if not quiet:
        print(f"\nSaved metrics: {metrics_csv}")
        print(f"Saved plot:    {plot_html}")

    session_meta = {
        "session_id": session_id,
        "Date": session_info.date_str,
        "Subject": session_info.subject_id,
        "BF_Type": session_info.bf_type,
    }
    return session_meta, missing_keys


def _list_sessions_from_raw() -> List[str]:
    raw_root = _data_dir() / "Raw"
    if not raw_root.exists():
        return []
    return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resting HR (Feedback) analysis end-to-end.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Session ID like 251216_TK")
    group.add_argument("--all-sessions", action="store_true", help="Process all sessions in Data/Raw")
    parser.add_argument("--rebuild-extracted", action="store_true", help="Delete and rebuild extracted CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    args = parser.parse_args()

    if args.all_sessions:
        sessions = _list_sessions_from_raw()
        if not sessions:
            print("No sessions found under Data/Raw")
            return 1
        failures: List[Tuple[str, str]] = []
        missing_rows: List[dict] = []
        results_root = Config().get_results_dir()
        report_path = results_root / "resting_hr_feedback_missing_segments.csv"

        for session_id in sessions:
            try:
                session_meta, missing_keys = run_session(
                    session_id=session_id,
                    rebuild_extracted=args.rebuild_extracted,
                    quiet=args.quiet,
                    allow_missing_segments=True,
                )
                for key in missing_keys:
                    row = dict(session_meta)
                    row["missing_segment_key"] = key
                    missing_rows.append(row)

                # Incremental write so partial runs still produce a report.
                pd.DataFrame(missing_rows).to_csv(report_path, index=False)
            except Exception as e:
                failures.append((session_id, str(e)))
                print(f"\n✗ ERROR session {session_id}: {e}\n")
                continue

        # Final write (also covers the empty case)
        pd.DataFrame(missing_rows).to_csv(report_path, index=False)
        if not args.quiet:
            print(f"Missing-segment report: {report_path}")

        if failures:
            print("=" * 70)
            print(f"Completed with failures: {len(failures)}/{len(sessions)}")
            for sid, msg in failures:
                print(f"- {sid}: {msg}")
            return 1

        return 0

        run_session(
        session_id=args.session,
        rebuild_extracted=args.rebuild_extracted,
        quiet=args.quiet,
        allow_missing_segments=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
