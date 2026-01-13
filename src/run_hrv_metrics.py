#!/usr/bin/env python3
"""
Step 2: HRV Analysis & Reporting

This script computes HRV metrics from R-peaks detected in Step 1:
1. Load peak information from JSON files
2. Generate RR intervals
3. Clean RR intervals (MAD method, physiological range filter)
4. Compute time-domain HRV metrics (SDNN, RMSSD, pNN50)
5. Compute frequency-domain HRV metrics (LF, HF, LF/HF ratio)
6. Export summary to CSV

Prerequisites:
    Run run_signal_diagnosis.py first to generate peaks_*.json files

Usage:
    python src/run_hrv_metrics.py --session 251216_TK
    python src/run_hrv_metrics.py --session 251216_TK --segment 03_Resting_HR_1_Set1
Output:
    Results/hrv_summaries/{session}_hrv_summary.csv  - Complete HRV metrics for all segments
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config
from analysis_pipeline.io_utils import (
    load_peaks_json,
    list_peaks_files,
    get_available_sessions,
    PeakInfo,
)
from analysis_pipeline.hrv import compute_hrv_metrics, HRVMetrics
import numpy as np


def process_peaks_file(
    peaks_path: Path,
    config: Config,
    verbose: bool = True,
) -> HRVMetrics:
    """
    Process a single peaks JSON file to compute HRV metrics.
    
    Parameters
    ----------
    peaks_path : Path
        Path to peaks JSON file.
    config : Config
        Pipeline configuration.
    verbose : bool
        Print progress messages.
    
    Returns
    -------
    HRVMetrics
        Computed HRV metrics.
    """
    # Load peak information
    peak_info = load_peaks_json(peaks_path)
    
    if verbose:
        print(f"\n  Segment: {peak_info.segment_name}")
        print(f"    Peaks: {len(peak_info.peak_indices)}")
        if peak_info.is_inverted:
            print(f"    Note: Signal was inverted (skewness={peak_info.skewness:.3f})")
    
    # Calculate duration
    duration_sec = peak_info.n_samples / peak_info.fs
    
    # Convert peak times to numpy array
    peak_times = np.array(peak_info.peak_times)
    
    # Compute HRV metrics
    # NOTE: Mean HR is computed on the central 30 seconds (when timestamps exist),
    # while diagnosis/peak detection stays unchanged.
    hrv_metrics = compute_hrv_metrics(
        peak_times=peak_times,
        segment_name=peak_info.segment_name,
        session_id=peak_info.session_id,
        duration_sec=duration_sec,
        config=config,
        peak_timestamps=peak_info.peak_timestamps,
        segment_start_timestamp=peak_info.segment_start_timestamp,
        segment_end_timestamp=peak_info.segment_end_timestamp,
    )
    
    if verbose:
        print(f"    Valid RR intervals: {hrv_metrics.n_valid_intervals}")
        print(f"    Removal rate: {hrv_metrics.removal_rate:.1%}")
        
        td = hrv_metrics.time_domain
        if not np.isnan(td.mean_hr):
            hr_label = "center 30s" if any("Mean HR (center 30s)" in n for n in hrv_metrics.quality_notes) else "full 60s"
            print(f"    Mean HR ({hr_label}): {td.mean_hr:.1f} bpm")
            print(f"    SDNN: {td.sdnn:.1f} ms")
            print(f"    RMSSD: {td.rmssd:.1f} ms")
            print(f"    pNN50: {td.pnn50:.1f}%")
        
        fd = hrv_metrics.frequency_domain
        if not np.isnan(fd.lf_hf_ratio):
            print(f"    LF/HF: {fd.lf_hf_ratio:.2f}")
    
    return hrv_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: HRV Analysis from Detected R-Peaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute HRV for all segments in a session
    python src/run_hrv_metrics.py --session 251216_TK
    
    # Compute HRV for a specific segment
    python src/run_hrv_metrics.py --session 251216_TK --segment 03_Resting_HR_1_Set1
    
    # List available sessions
    python src/run_hrv_metrics.py --list-sessions
        """
    )
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        help="Session ID (e.g., 251216_TK)"
    )
    
    parser.add_argument(
        "--segment", "-g",
        type=str,
        default=None,
        help="Specific segment to process (without peaks_ prefix and .json extension)"
    )
    
    parser.add_argument(
        "--list-sessions", "-l",
        action="store_true",
        help="List available sessions and exit"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # List sessions mode
    if args.list_sessions:
        sessions = get_available_sessions(config)
        if sessions:
            print("Available sessions with peaks data:")
            for session in sessions:
                try:
                    peaks_files = list_peaks_files(session, config)
                    if peaks_files:
                        print(f"  - {session} ({len(peaks_files)} peak files)")
                except Exception:
                    pass
        else:
            print("No sessions found.")
        return 0
    
    # Validate session argument
    if not args.session:
        parser.error("--session is required (use --list-sessions to see available)")
    
    session_id = args.session
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("HRV Analysis Pipeline - Step 2")
        print("=" * 60)
        print(f"Session: {session_id}")
    
    # Get peaks files
    try:
        peaks_files = list_peaks_files(session_id, config)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    if not peaks_files:
        print(f"Error: No peaks files found for session '{session_id}'.")
        print("Please run Step 1 first:")
        print(f"  python src/run_signal_diagnosis.py --session {session_id}")
        return 1
    
    # Filter to specific segment if requested
    if args.segment:
        target_name = f"{config.PEAKS_PREFIX}{args.segment}.json"
        peaks_files = [f for f in peaks_files if f.name == target_name]
        if not peaks_files:
            print(f"Error: Peaks file for segment '{args.segment}' not found.")
            print(f"Available segments:")
            for f in list_peaks_files(session_id, config):
                segment_name = f.stem.replace(config.PEAKS_PREFIX, "")
                print(f"  - {segment_name}")
            return 1
    
    if verbose:
        print(f"Segments to analyze: {len(peaks_files)}")
    
    # Process each peaks file
    all_metrics: List[HRVMetrics] = []
    
    for peaks_path in peaks_files:
        try:
            metrics = process_peaks_file(peaks_path, config, verbose)
            all_metrics.append(metrics)
        except Exception as e:
            segment_name = peaks_path.stem.replace(config.PEAKS_PREFIX, "")
            print(f"  ✗ ERROR processing {segment_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_metrics:
        print("Error: No segments were successfully processed.")
        return 1
    
    # Create summary DataFrame
    summary_data = [m.to_flat_dict() for m in all_metrics]
    df = pd.DataFrame(summary_data)
    
    # Reorder columns for readability
    column_order = [
        "segment_name", "session_id", "duration_sec", 
        "n_beats", "n_valid_intervals", "removal_rate",
        "time_mean_rr", "time_sdnn", "time_rmssd", "time_pnn50",
        "time_mean_hr", "time_median_hr", "time_std_hr", "time_min_hr", "time_max_hr",
        "freq_vlf_power", "freq_lf_power", "freq_hf_power", "freq_total_power",
        "freq_lf_norm", "freq_hf_norm", "freq_lf_hf_ratio",
        "quality_notes",
    ]
    
    # Only include columns that exist
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]
    
    # Save to CSV
    output_path = config.get_hrv_summary_path(session_id)
    df.to_csv(output_path, index=False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Segments analyzed: {len(all_metrics)}")
        print(f"\nOutput saved to: {output_path}")
        
        # Print summary table
        print("\n" + "-" * 60)
        print("HRV Summary (Time Domain)")
        print("-" * 60)
        
        summary_cols = ["segment_name", "time_mean_hr", "time_median_hr", "time_sdnn", "time_rmssd", "time_pnn50"]
        available_cols = [c for c in summary_cols if c in df.columns]
        
        if available_cols:
            print(df[available_cols].to_string(index=False))
        
        print("\n" + "-" * 60)
        print("HRV Summary (Frequency Domain)")
        print("-" * 60)
        
        freq_cols = ["segment_name", "freq_lf_power", "freq_hf_power", "freq_lf_hf_ratio"]
        available_freq_cols = [c for c in freq_cols if c in df.columns]
        
        if available_freq_cols:
            print(df[available_freq_cols].to_string(index=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
