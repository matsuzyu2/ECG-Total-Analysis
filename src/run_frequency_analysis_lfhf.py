#!/usr/bin/env python3
"""Run HRV frequency analysis (AR/Burg) per subject/session.

This script reads existing `Results/{session}/peaks_*.json` files and produces:
- `Results/{session}/hrv_frequency_ar_metrics.csv`
- `Results/{session}/hrv_psd_<segment>.html` (LF/HF annotated)

Why AR?
For ~60s windows, Welch/FFT can be unstable. AR (Burg) tends to give smoother PSD
and more robust LF/HF estimates when combined with linear detrending.

Usage:
  python src/run_frequency_analysis_lfhf.py --session 251224_HT
  python src/run_frequency_analysis_lfhf.py --all-sessions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config
from analysis_pipeline.io_utils import list_peaks_files, load_peaks_json, get_available_sessions
from analysis_pipeline.hrv import clean_rr_intervals
from analysis_pipeline.hrv_frequency_ar import (
    compute_hrv_psd_ar,
    hrv_psd_html_fragment,
    write_hrv_psd_summary_html,
)


def _process_session(session_id: str, *, config: Config, quiet: bool) -> Path:
    peaks_files = list_peaks_files(session_id, config)
    if not peaks_files:
        raise FileNotFoundError(f"No peaks files found for session '{session_id}'. Run Step 1 first.")

    rows: List[dict] = []
    results_dir = config.get_session_results_dir(session_id)

    summary_psd_html_name = "hrv_psd_summary.html"
    psd_fragments: List[tuple[str, str]] = []
    _plotlyjs_mode = "cdn"  # include plotly.js only once

    for peaks_path in peaks_files:
        peak_info = load_peaks_json(peaks_path)
        segment_name = peak_info.segment_name

        peak_times = np.asarray(peak_info.peak_times, dtype=float)
        rr_ms = np.diff(peak_times) * 1000.0
        rr_cleaned = clean_rr_intervals(rr_ms, config).rr_cleaned

        psd_result = compute_hrv_psd_ar(
            rr_cleaned,
            config=config,
            fs_resample_hz=4.0,
            ar_order=16,
            nfft=1024,
        )

        fragment = hrv_psd_html_fragment(
            psd_result,
            title=f"HRV PSD (AR/Burg) - {session_id} / {segment_name}",
            config=config,
            include_plotlyjs=_plotlyjs_mode,
        )
        psd_fragments.append((segment_name, fragment))
        _plotlyjs_mode = False

        rows.append(
            {
                "session_id": session_id,
                "segment_name": segment_name,
                "method": psd_result.method,
                "ar_order": psd_result.ar_order,
                "nfft": psd_result.nfft,
                "fs_resample_hz": psd_result.fs_resample_hz,
                "vlf_power": psd_result.vlf_power,
                "lf_power": psd_result.lf_power,
                "hf_power": psd_result.hf_power,
                "total_power": psd_result.total_power,
                "lf_norm": psd_result.lf_norm,
                "hf_norm": psd_result.hf_norm,
                "lf_hf_ratio": psd_result.lf_hf_ratio,
                "psd_plot": summary_psd_html_name,
                "notes": "; ".join(psd_result.notes) if psd_result.notes else "",
            }
        )

        if not quiet:
            print(f"✓ {session_id} / {segment_name}: LF/HF={psd_result.lf_hf_ratio:.2f}")

    # Write a single combined PSD page
    if psd_fragments:
        write_hrv_psd_summary_html(
            session_id=session_id,
            items=psd_fragments,
            output_html=results_dir / summary_psd_html_name,
        )

    out_csv = results_dir / "hrv_frequency_ar_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="HRV frequency analysis per session (AR/Burg).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", type=str, help="Session ID like 251224_HT")
    group.add_argument("--all-sessions", action="store_true", help="Process all sessions with peaks")
    parser.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = parser.parse_args()

    config = Config()
    quiet = args.quiet

    if args.all_sessions:
        sessions = get_available_sessions(config)
        if not sessions:
            print("No sessions found.")
            return 1

        failures = []
        for sid in sessions:
            try:
                out = _process_session(sid, config=config, quiet=quiet)
                if not quiet:
                    print(f"Saved: {out}")
            except Exception as e:
                failures.append((sid, str(e)))
                print(f"✗ {sid}: {e}")
                continue

        return 1 if failures else 0

    out_csv = _process_session(args.session, config=config, quiet=quiet)
    if not quiet:
        print(f"Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
