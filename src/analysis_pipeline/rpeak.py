"""R-peak detection.
Uses Savitzky–Golay derivatives method
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass

import numpy as np

from .config import Config, default_config


@dataclass
class DetectionResult:
    """Result of R-peak detection."""
    peak_indices: np.ndarray    # Sample indices of detected R-peaks
    peak_times: np.ndarray      # Times in seconds
    method_used: str            # Detection method that was used
    quality_score: float        # 0-1 quality estimate
    quality_notes: List[str]    # Quality warnings/notes
    
    @property
    def n_peaks(self) -> int:
        """Number of detected peaks."""
        return len(self.peak_indices)
    
    def get_rr_intervals_ms(self) -> np.ndarray:
        """Get RR intervals in milliseconds."""
        if len(self.peak_times) < 2:
            return np.array([])
        return np.diff(self.peak_times) * 1000


def detect_rpeaks(
    ecg: np.ndarray,
    fs: Optional[int] = None,
    method: Optional[str] = None,
    time: Optional[np.ndarray] = None,
    config: Config = default_config,
) -> DetectionResult:
    """Detect R-peaks in ECG signal.

    Uses a single method: windowed Savitzky–Golay derivative detection.

    Parameters
    ----------
    ecg : np.ndarray
        Filtered ECG signal (should be in mV, correctly oriented).
    fs : int, optional
        Sampling frequency in Hz. Defaults to config.SAMPLING_RATE.
    method : str, optional
        Detection method. Defaults to config.RPEAK_METHOD.
    time : np.ndarray, optional
        Time array in seconds. If None, generated from fs.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    DetectionResult
        Detection results with peaks and quality metrics.
    """
    if fs is None:
        fs = config.SAMPLING_RATE
    if method is None:
        method = config.RPEAK_METHOD
    if time is None:
        time = np.arange(len(ecg)) / fs

    quality_notes: List[str] = []
    base_score = 1.0

    method_used = "sgolay_derivative"
    if method is not None and method != "sgolay_derivative":
        quality_notes.append(
            f"⚠ method='{method}' is ignored (only 'sgolay_derivative' is supported)"
        )
        base_score -= 0.05

    peak_indices = _sgolay_derivative_peak_detection(
        ecg,
        fs,
        window_sec=30.0,
        smooth=True,
        smooth_window=35,
        smooth_polyorder=3,
        deriv_window=11,
        deriv_polyorder=2,
        th_mult=2.5,
        min_distance_ms=float(config.RR_MIN_MS),
    )

    if peak_indices is None or len(peak_indices) == 0:
        quality_notes.append("⚠ No peaks detected by sgolay_derivative")
        base_score -= 0.3
    
    # Normalize/clean indices
    peak_indices = np.asarray(peak_indices, dtype=int)
    peak_indices = np.unique(peak_indices)
    peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(ecg))]

    # Small alignment fix: snap to local maxima
    if len(peak_indices) > 0:
        peak_indices = refine_peaks(ecg, peak_indices, fs, window_ms=50)
        peak_indices = np.unique(peak_indices)
    
    # Calculate peak times
    peak_times = time[peak_indices] if len(peak_indices) > 0 else np.array([])
    
    # Quality assessment
    assessed_score, additional_notes = _assess_detection_quality(
        peak_indices, ecg, fs, config
    )
    quality_notes.extend(additional_notes)

    # Combine with basic penalties (e.g., unsupported method provided)
    quality_score = max(0.0, min(1.0, assessed_score * max(0.0, base_score)))
    
    return DetectionResult(
        peak_indices=peak_indices,
        peak_times=peak_times,
        method_used=method_used,
        quality_score=quality_score,
        quality_notes=quality_notes,
    )


def _sgolay_derivative_peak_detection(
    ecg: np.ndarray,
    fs: int,
    *,
    window_sec: float = 30.0,
    smooth: bool = True,
    smooth_window: int = 35,
    smooth_polyorder: int = 3,
    deriv_window: int = 11,
    deriv_polyorder: int = 2,
    th_mult: float = 3.0,
    min_distance_ms: float = 300.0,
) -> np.ndarray:
    """Savitzky–Golay derivative peak detector (Analysis.R-like)."""
    from scipy.signal import savgol_filter

    if ecg.size == 0:
        return np.array([], dtype=int)

    def _odd_leq(n: int, upper: int) -> int:
        n2 = min(n, upper)
        if n2 % 2 == 0:
            n2 -= 1
        return n2

    # Global smoothing (like Analysis.R's ECG_smt)
    x = np.asarray(ecg, dtype=float)
    if smooth:
        w = _odd_leq(int(smooth_window), int(len(x)))
        if w >= smooth_polyorder + 2 and w >= 5:
            x = savgol_filter(x, window_length=w, polyorder=smooth_polyorder, deriv=0)

    win_samples = max(1, int(round(window_sec * fs)))
    min_distance = max(1, int(round(min_distance_ms * fs / 1000.0)))

    candidates: List[int] = []
    for start in range(0, len(x), win_samples):
        end = min(len(x), start + win_samples)
        xw = x[start:end]
        if len(xw) < 5:
            continue

        dw = _odd_leq(int(deriv_window), int(len(xw)))
        if dw < deriv_polyorder + 2 or dw < 5:
            continue

        d1 = savgol_filter(xw, window_length=dw, polyorder=deriv_polyorder, deriv=1, delta=1.0)
        d2 = savgol_filter(xw, window_length=dw, polyorder=deriv_polyorder, deriv=2, delta=1.0)

        th = float(np.std(d2)) * float(th_mult)
        if not np.isfinite(th) or th <= 0:
            continue

        # d1: + -> - crossing, and d2 strongly negative
        idx = np.where((d1[:-1] > 0) & (d1[1:] < 0) & (d2[1:] < -th))[0] + 1
        if idx.size:
            candidates.extend((start + idx).tolist())

    if not candidates:
        return np.array([], dtype=int)

    # Refractory period: keep the stronger peak when too close
    candidates = sorted(set(int(i) for i in candidates if 0 <= i < len(ecg)))
    kept: List[int] = []
    for idx in candidates:
        if not kept:
            kept.append(idx)
            continue

        if idx - kept[-1] >= min_distance:
            kept.append(idx)
            continue

        # Too close: keep the higher amplitude
        if x[idx] > x[kept[-1]]:
            kept[-1] = idx

    return np.asarray(kept, dtype=int)


def _assess_detection_quality(
    peak_indices: np.ndarray,
    ecg: np.ndarray,
    fs: int,
    config: Config,
) -> Tuple[float, List[str]]:
    """
    Assess quality of R-peak detection.
    
    Parameters
    ----------
    peak_indices : np.ndarray
        Detected peak indices.
    ecg : np.ndarray
        ECG signal.
    fs : int
        Sampling frequency.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    Tuple[float, List[str]]
        (quality_score, quality_notes)
    """
    quality_score = 1.0
    notes = []
    
    if len(peak_indices) == 0:
        return 0.0, ["✗ No R-peaks detected"]
    
    # Calculate expected number of peaks
    duration_sec = len(ecg) / fs
    
    # Expected HR range: 40-180 bpm
    min_expected = int(duration_sec * 40 / 60)
    max_expected = int(duration_sec * 180 / 60)
    
    n_peaks = len(peak_indices)
    
    if n_peaks < min_expected:
        quality_score -= 0.3
        notes.append(f"⚠ Too few peaks detected ({n_peaks} < {min_expected} expected for 40bpm)")
    elif n_peaks > max_expected:
        quality_score -= 0.2
        notes.append(f"⚠ Too many peaks detected ({n_peaks} > {max_expected} expected for 180bpm)")
    else:
        estimated_hr = n_peaks / duration_sec * 60
        notes.append(f"✓ Peak count reasonable (estimated HR: {estimated_hr:.0f} bpm)")
    
    # Check RR interval variability
    if len(peak_indices) >= 2:
        rr_intervals = np.diff(peak_indices) / fs * 1000  # in ms
        
        # Check for physiologically impossible intervals
        too_short = np.sum(rr_intervals < config.RR_MIN_MS)
        too_long = np.sum(rr_intervals > config.RR_MAX_MS)
        
        if too_short > 0:
            quality_score -= 0.1 * (too_short / len(rr_intervals))
            notes.append(f"⚠ {too_short} RR intervals < {config.RR_MIN_MS}ms (too short)")
        
        if too_long > 0:
            quality_score -= 0.1 * (too_long / len(rr_intervals))
            notes.append(f"⚠ {too_long} RR intervals > {config.RR_MAX_MS}ms (too long)")
        
        # Check coefficient of variation
        cv = np.std(rr_intervals) / np.mean(rr_intervals)
        if cv > 0.5:
            quality_score -= 0.1
            notes.append(f"⚠ High RR variability (CV={cv:.2f}). May indicate detection errors.")
    
    if not notes:
        notes.append("✓ Detection quality appears good")
    
    return max(0.0, quality_score), notes


def refine_peaks(
    ecg: np.ndarray,
    peak_indices: np.ndarray,
    fs: int,
    window_ms: float = 50,
) -> np.ndarray:
    """
    Refine peak positions to local maximum within window.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    peak_indices : np.ndarray
        Initial peak indices.
    fs : int
        Sampling frequency.
    window_ms : float
        Search window in milliseconds.
    
    Returns
    -------
    np.ndarray
        Refined peak indices.
    """
    window_samples = int(window_ms * fs / 1000)
    refined = np.zeros_like(peak_indices)
    
    for i, idx in enumerate(peak_indices):
        start = max(0, idx - window_samples)
        end = min(len(ecg), idx + window_samples + 1)
        local_max = np.argmax(ecg[start:end])
        refined[i] = start + local_max
    
    return refined
