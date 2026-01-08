#!/usr/bin/env python3
"""
Step 1: Signal Profiling & Robust R-Peak Detection

This script performs a comprehensive check of ECG signals:
1. Load segment data with automatic µV→mV conversion
2. Signal profiling (EDA):
   - Polarity detection
   - Power spectral density analysis 
3. Bandpass filtering (0.5-40 Hz, zero-phase)
4. R-peak detection
5. Generate interactive HTML diagnostic reports

Usage:
    python src/run_signal_diagnosis.py --session 251216_TK
    python src/run_signal_diagnosis.py --session 251216_TK --segment 03_Resting_HR_1_Set1

Output:
    Results/{session}/diagnosis_{segment}.html  - Interactive diagnostic report
    Results/{session}/peaks_{segment}.json      - Peak detection results
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure this script works when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pipeline.config import Config
from analysis_pipeline.io_utils import (
    load_segment_csv,
    save_peaks_json,
    list_segment_files,
    get_available_sessions,
    PeakInfo,
)
from analysis_pipeline.diagnosis import (
    check_polarity,
    auto_invert,
    compute_psd,
    create_diagnosis_report,
)
from analysis_pipeline.preprocess import filter_ecg
from analysis_pipeline.rpeak import detect_rpeaks


def process_segment(
    segment_path: Path,
    session_id: str,
    config: Config,
    verbose: bool = True,
) -> bool:
    """
    Process a single ECG segment through the diagnosis pipeline.
    
    Parameters
    ----------
    segment_path : Path
        Path to segment CSV file.
    session_id : str
        Session identifier.
    config : Config
        Pipeline configuration.
    verbose : bool
        Print progress messages.
    
    Returns
    -------
    bool
        True if processing succeeded.
    """
    segment_name = segment_path.stem
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {segment_name}")
        print(f"{'='*60}")
    
    try:
        # =====================================================================
        # Step 1: Load data with automatic unit conversion
        # =====================================================================
        if verbose:
            print("  [1/5] Loading data...")
        
        segment_data = load_segment_csv(segment_path, session_id, config)
        
        if verbose:
            conversion_status = "µV→mV applied" if segment_data.was_converted else "already in mV"
            print(f"        ✓ Loaded {segment_data.n_samples:,} samples ({segment_data.duration_seconds:.1f}s)")
            print(f"        ✓ Unit conversion: {conversion_status}")
            print(f"        ✓ Mean |ECG|: {segment_data.original_mean_abs:.2f} (before conversion)")
        
        ecg_raw = segment_data.ecg.copy()
        time = segment_data.time
        
        # =====================================================================
        # Step 2: Signal profiling - Polarity check
        # =====================================================================
        if verbose:
            print("  [2/5] Analyzing signal polarity...")
        
        polarity_result = check_polarity(ecg_raw, config)

        # NOTE: Automatic inversion is intentionally disabled.
        # We keep polarity/skewness calculation for diagnostics, but do not
        # modify the signal based on it.
        ecg_oriented = ecg_raw
        was_inverted = False
        
        if verbose:
            print(f"        ✓ Skewness: {polarity_result.skewness:.3f}")
            if polarity_result.is_inverted:
                print(f"        ⚠ Negative skewness detected (possible inverted polarity)")
                print(f"        ⚠ Auto-inversion is DISABLED; signal not modified")
            else:
                print(f"        ✓ Polarity OK (no inversion needed)")
        
        # =====================================================================
        # Step 3: Bandpass filtering
        # =====================================================================
        if verbose:
            print("  [3/5] Applying bandpass filter...")
            if segment_data.has_padding:
                print(f"        ℹ Segment has {config.FILTER_PADDING_SEC}s padding for edge artifact handling")
        
        ecg_filtered = filter_ecg(ecg_oriented, config=config)
        
        if verbose:
            print(f"        ✓ Butterworth {config.BANDPASS_LOW}-{config.BANDPASS_HIGH}Hz (order={config.FILTER_ORDER})")
        
        # =====================================================================
        # Step 3.5: Trim padding after filtering (if present)
        # =====================================================================
        if segment_data.has_padding:
            if verbose:
                print(f"        ℹ Trimming padding: keeping indices [{segment_data.original_start_idx}:{segment_data.original_end_idx+1}]")
            
            # Validate bounds before slicing
            start_idx = segment_data.original_start_idx
            end_idx = segment_data.original_end_idx + 1
            
            if start_idx < 0 or end_idx > len(ecg_filtered) or start_idx >= end_idx:
                print(f"        ⚠ Warning: Invalid padding bounds [{start_idx}:{end_idx}] for array length {len(ecg_filtered)}")
                print(f"        ⚠ Skipping padding trim, using full signal")
            else:
                # Trim the filtered signal to remove padding
                ecg_filtered = ecg_filtered[start_idx:end_idx]
                
                # Also trim the raw signal and time array for consistency
                ecg_raw = ecg_raw[start_idx:end_idx]
                ecg_oriented = ecg_oriented[start_idx:end_idx]
                time = time[start_idx:end_idx]
                
                if verbose:
                    print(f"        ✓ Trimmed to {len(ecg_filtered):,} samples ({len(ecg_filtered)/segment_data.fs:.1f}s)")
        
        # =====================================================================
        # Step 4: Power spectral density analysis
        # =====================================================================
        if verbose:
            print("  [4/5] Computing power spectrum...")
        
        psd_result = compute_psd(ecg_oriented, segment_data.fs, config)
        
        if verbose:
            for note in psd_result.noise_notes:
                print(f"        {note}")
        
        # =====================================================================
        # Step 5: R-peak detection
        # =====================================================================
        if verbose:
            print("  [5/5] Detecting R-peaks...")
        
        detection_result = detect_rpeaks(
            ecg_filtered,
            fs=segment_data.fs,
            time=time,
            config=config,
        )
        
        if verbose:
            print(f"        ✓ Method: {detection_result.method_used}")
            print(f"        ✓ Peaks detected: {detection_result.n_peaks}")
            print(f"        ✓ Quality score: {detection_result.quality_score:.2f}")
            for note in detection_result.quality_notes:
                print(f"        {note}")
        
        # =====================================================================
        # Generate diagnostic report (HTML)
        # =====================================================================
        if verbose:
            print("  Generating diagnostic report...")
        
        html_content = create_diagnosis_report(
            ecg_raw=ecg_raw,  # Pre-inversion, pre-filter (original polarity)
            ecg_filtered=ecg_filtered,
            time=time,
            peak_indices=detection_result.peak_indices,
            polarity_result=polarity_result,
            psd_result=psd_result,
            segment_name=segment_name,
            session_id=session_id,
            was_converted=segment_data.was_converted,
            was_inverted=was_inverted,
            fs=segment_data.fs,
            config=config,
        )
        
        # Save HTML report
        html_path = config.get_diagnosis_path(session_id, segment_name)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if verbose:
            print(f"        ✓ Saved: {html_path.name}")
        
        # =====================================================================
        # Save peak information (JSON)
        # =====================================================================
        quality_notes = detection_result.quality_notes.copy()
        quality_notes.extend(psd_result.noise_notes)

        peak_timestamps = None
        segment_start_timestamp = None
        segment_end_timestamp = None

        if segment_data.timestamps is not None:
            ts = segment_data.timestamps

            def _to_optional_str(value):
                if value is None:
                    return None
                text = str(value)
                return None if text.strip().lower() in {"<na>", "nan", "nat", "none"} else text

            # Segment bounds
            if len(ts) > 0:
                segment_start_timestamp = _to_optional_str(ts[0])
                segment_end_timestamp = _to_optional_str(ts[-1])

            # Peak timestamps
            peak_timestamps = []
            for raw_idx in detection_result.peak_indices:
                idx = int(raw_idx)
                if 0 <= idx < len(ts):
                    peak_timestamps.append(_to_optional_str(ts[idx]))
                else:
                    peak_timestamps.append(None)
                    quality_notes.append("⚠ Peak index out of range for Timestamp mapping")
        else:
            quality_notes.append("⚠ Timestamp column not found; peak_timestamps not saved")
        
        peak_info = PeakInfo(
            segment_name=segment_name,
            session_id=session_id,
            peak_indices=detection_result.peak_indices.tolist(),
            peak_times=detection_result.peak_times.tolist(),
            peak_timestamps=peak_timestamps,
            segment_start_timestamp=segment_start_timestamp,
            segment_end_timestamp=segment_end_timestamp,
            is_inverted=was_inverted,
            skewness=float(polarity_result.skewness),
            was_converted=segment_data.was_converted,
            fs=segment_data.fs,
            n_samples=segment_data.n_samples,
            detection_method=detection_result.method_used,
            processed_at=datetime.now().isoformat(),
            quality_notes=quality_notes,
        )
        
        peaks_path = config.get_peaks_path(session_id, segment_name)
        save_peaks_json(peak_info, peaks_path)
        
        if verbose:
            print(f"        ✓ Saved: {peaks_path.name}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR processing {segment_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: ECG Signal Diagnosis and R-Peak Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all segments for a session
    python src/run_signal_diagnosis.py --session 251216_TK
    
    # Process a specific segment
    python src/run_signal_diagnosis.py --session 251216_TK --segment 03_Resting_HR_1_Set1
    
    # List available sessions
    python src/run_signal_diagnosis.py --list-sessions
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
        help="Specific segment to process (without .csv extension)"
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
            print("Available sessions:")
            for session in sessions:
                segments_dir = config.get_segments_dir(session)
                n_segments = len(list(segments_dir.glob("*.csv")))
                print(f"  - {session} ({n_segments} segments)")
        else:
            print("No sessions found with processed segments.")
        return 0
    
    # Validate session argument
    if not args.session:
        parser.error("--session is required (use --list-sessions to see available)")
    
    session_id = args.session
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("ECG Signal Diagnosis Pipeline - Step 1")
        print("=" * 60)
        print(f"Session: {session_id}")
        print(f"Config: Fs={config.SAMPLING_RATE}Hz, "
              f"Filter={config.BANDPASS_LOW}-{config.BANDPASS_HIGH}Hz")
        print(f"Polarity threshold: skewness < {config.SKEWNESS_THRESHOLD}")
    
    # Get segment files
    try:
        segment_files = list_segment_files(session_id, config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Filter to specific segment if requested
    if args.segment:
        segment_files = [f for f in segment_files if f.stem == args.segment]
        if not segment_files:
            print(f"Error: Segment '{args.segment}' not found.")
            print(f"Available segments in {session_id}:")
            for f in list_segment_files(session_id, config):
                print(f"  - {f.stem}")
            return 1
    
    if verbose:
        print(f"Segments to process: {len(segment_files)}")
    
    # Process each segment
    results = []
    for segment_path in segment_files:
        success = process_segment(segment_path, session_id, config, verbose)
        results.append((segment_path.stem, success))
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for _, s in results if s)
        failed = len(results) - successful
        
        print(f"Total segments: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed segments:")
            for name, success in results:
                if not success:
                    print(f"  - {name}")
        
        results_dir = config.get_session_results_dir(session_id)
        print(f"\nOutput directory: {results_dir}")
        print("\nNext step: Review diagnosis_*.html reports, then run:")
        print(f"  python src/run_hrv_metrics.py --session {session_id}")
    
    return 0 if all(s for _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
