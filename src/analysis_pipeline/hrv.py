"""
HRV (Heart Rate Variability) analysis module.

Provides:
- RR interval generation and cleaning (MAD method, physiological range)
- Time-domain HRV metrics (SDNN, RMSSD, pNN50)
- Frequency-domain HRV metrics (LF, HF, LF/HF ratio)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import warnings

import numpy as np
from scipy import signal
from scipy import interpolate

from .config import Config, default_config


@dataclass
class RRCleaningResult:
    """Result of RR interval cleaning."""
    rr_original: np.ndarray      # Original RR intervals (ms)
    rr_cleaned: np.ndarray       # Cleaned RR intervals (ms)
    n_removed_physiological: int  # Removed by physiological range filter
    n_removed_mad: int           # Removed by MAD outlier detection
    removal_rate: float          # Percentage of intervals removed


@dataclass
class TimeDomainHRV:
    """Time-domain HRV metrics."""
    mean_rr: float    # Mean RR interval (ms)
    sdnn: float       # Standard deviation of NN intervals (ms)
    rmssd: float      # Root mean square of successive differences (ms)
    pnn50: float      # Percentage of successive intervals > 50ms (%)
    mean_hr: float    # Mean heart rate (bpm)
    std_hr: float     # Standard deviation of heart rate (bpm)
    min_hr: float     # Minimum heart rate (bpm)
    max_hr: float     # Maximum heart rate (bpm)
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FrequencyDomainHRV:
    """Frequency-domain HRV metrics."""
    vlf_power: float    # VLF power (ms²)
    lf_power: float     # LF power (ms²)
    hf_power: float     # HF power (ms²)
    total_power: float  # Total power (ms²)
    lf_norm: float      # LF normalized units (%)
    hf_norm: float      # HF normalized units (%)
    lf_hf_ratio: float  # LF/HF ratio
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class HRVMetrics:
    """Complete HRV metrics."""
    segment_name: str
    session_id: str
    duration_sec: float
    n_beats: int
    n_valid_intervals: int
    removal_rate: float
    time_domain: TimeDomainHRV
    frequency_domain: FrequencyDomainHRV
    quality_notes: List[str]
    
    def to_flat_dict(self) -> Dict[str, any]:
        """Convert to flat dictionary for CSV export."""
        result = {
            "segment_name": self.segment_name,
            "session_id": self.session_id,
            "duration_sec": self.duration_sec,
            "n_beats": self.n_beats,
            "n_valid_intervals": self.n_valid_intervals,
            "removal_rate": self.removal_rate,
        }
        result.update({f"time_{k}": v for k, v in self.time_domain.to_dict().items()})
        result.update({f"freq_{k}": v for k, v in self.frequency_domain.to_dict().items()})
        result["quality_notes"] = "; ".join(self.quality_notes)
        return result


def clean_rr_intervals(
    rr_intervals: np.ndarray,
    config: Config = default_config,
) -> RRCleaningResult:
    """
    Clean RR intervals using physiological range and MAD outlier detection.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in milliseconds.
    config : Config
        Pipeline configuration with RR_MIN_MS, RR_MAX_MS, MAD_THRESHOLD.
    
    Returns
    -------
    RRCleaningResult
        Cleaned intervals with removal statistics.
    """
    if len(rr_intervals) == 0:
        return RRCleaningResult(
            rr_original=rr_intervals,
            rr_cleaned=rr_intervals,
            n_removed_physiological=0,
            n_removed_mad=0,
            removal_rate=0.0,
        )
    
    original_count = len(rr_intervals)
    
    # Step 1: Physiological range filter
    physio_mask = (rr_intervals >= config.RR_MIN_MS) & (rr_intervals <= config.RR_MAX_MS)
    rr_physio = rr_intervals[physio_mask]
    n_removed_physio = original_count - len(rr_physio)
    
    if len(rr_physio) == 0:
        return RRCleaningResult(
            rr_original=rr_intervals,
            rr_cleaned=np.array([]),
            n_removed_physiological=n_removed_physio,
            n_removed_mad=0,
            removal_rate=1.0,
        )
    
    # Step 2: MAD-based outlier removal
    median_rr = np.median(rr_physio)
    mad = np.median(np.abs(rr_physio - median_rr))
    
    # Avoid division by zero
    if mad < 1e-10:
        mad = np.std(rr_physio) * 0.6745  # Fallback to scaled std
    
    # Calculate z-scores using MAD
    mad_scores = np.abs(rr_physio - median_rr) / (mad * 1.4826)  # 1.4826 for normal consistency
    mad_mask = mad_scores < config.MAD_THRESHOLD
    
    rr_cleaned = rr_physio[mad_mask]
    n_removed_mad = len(rr_physio) - len(rr_cleaned)
    
    total_removed = n_removed_physio + n_removed_mad
    removal_rate = total_removed / original_count if original_count > 0 else 0.0
    
    return RRCleaningResult(
        rr_original=rr_intervals,
        rr_cleaned=rr_cleaned,
        n_removed_physiological=n_removed_physio,
        n_removed_mad=n_removed_mad,
        removal_rate=removal_rate,
    )


def compute_time_domain(rr_intervals: np.ndarray) -> TimeDomainHRV:
    """
    Compute time-domain HRV metrics.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        Cleaned RR intervals in milliseconds.
    
    Returns
    -------
    TimeDomainHRV
        Time-domain HRV metrics.
    """
    if len(rr_intervals) < 2:
        return TimeDomainHRV(
            mean_rr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan,
            mean_hr=np.nan, std_hr=np.nan, min_hr=np.nan, max_hr=np.nan,
        )
    
    # Basic statistics
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)  # Sample std
    
    # Successive differences
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # pNN50: percentage of successive differences > 50ms
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
    
    # Heart rate metrics (bpm = 60000 / RR in ms)
    hr = 60000 / rr_intervals
    mean_hr = np.mean(hr)
    std_hr = np.std(hr, ddof=1)
    min_hr = np.min(hr)
    max_hr = np.max(hr)
    
    return TimeDomainHRV(
        mean_rr=mean_rr,
        sdnn=sdnn,
        rmssd=rmssd,
        pnn50=pnn50,
        mean_hr=mean_hr,
        std_hr=std_hr,
        min_hr=min_hr,
        max_hr=max_hr,
    )


def compute_frequency_domain(
    rr_intervals: np.ndarray,
    config: Config = default_config,
) -> FrequencyDomainHRV:
    """
    Compute frequency-domain HRV metrics using Welch's method.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        Cleaned RR intervals in milliseconds.
    config : Config
        Pipeline configuration with frequency band definitions.
    
    Returns
    -------
    FrequencyDomainHRV
        Frequency-domain HRV metrics.
    """
    if len(rr_intervals) < 10:
        return FrequencyDomainHRV(
            vlf_power=np.nan, lf_power=np.nan, hf_power=np.nan,
            total_power=np.nan, lf_norm=np.nan, hf_norm=np.nan,
            lf_hf_ratio=np.nan,
        )
    
    # Create evenly sampled RR series using interpolation
    # Cumulative time of R-peaks
    rr_seconds = rr_intervals / 1000  # Convert to seconds
    t_rr = np.cumsum(rr_seconds)
    t_rr = np.insert(t_rr, 0, 0)[:-1]  # Time at each beat
    
    # Resample at 4 Hz (standard for HRV frequency analysis)
    fs_resample = 4.0
    t_interp = np.arange(t_rr[0], t_rr[-1], 1/fs_resample)
    
    if len(t_interp) < 10:
        return FrequencyDomainHRV(
            vlf_power=np.nan, lf_power=np.nan, hf_power=np.nan,
            total_power=np.nan, lf_norm=np.nan, hf_norm=np.nan,
            lf_hf_ratio=np.nan,
        )
    
    # Cubic spline interpolation
    try:
        interp_func = interpolate.interp1d(t_rr, rr_intervals, kind='cubic', 
                                           fill_value='extrapolate')
        rr_interp = interp_func(t_interp)
    except Exception:
        # Fallback to linear interpolation
        interp_func = interpolate.interp1d(t_rr, rr_intervals, kind='linear',
                                           fill_value='extrapolate')
        rr_interp = interp_func(t_interp)
    
    # Detrend
    rr_detrend = rr_interp - np.mean(rr_interp)
    
    # Compute PSD using Welch's method
    nperseg = min(config.HRV_NPERSEG, len(rr_detrend))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frequencies, psd = signal.welch(
            rr_detrend, 
            fs=fs_resample, 
            nperseg=nperseg,
            scaling='density'
        )
    
    # Calculate band powers
    def band_power(freq_range: Tuple[float, float]) -> float:
        mask = (frequencies >= freq_range[0]) & (frequencies < freq_range[1])
        if not np.any(mask):
            return 0.0
        # Integrate using trapezoidal rule (numpy 2.0+ uses trapezoid)
        try:
            return np.trapezoid(psd[mask], frequencies[mask])
        except AttributeError:
            return np.trapz(psd[mask], frequencies[mask])
    
    vlf_power = band_power(config.VLF_BAND)
    lf_power = band_power(config.LF_BAND)
    hf_power = band_power(config.HF_BAND)
    total_power = vlf_power + lf_power + hf_power
    
    # Normalized units (excluding VLF)
    lf_hf_sum = lf_power + hf_power
    if lf_hf_sum > 0:
        lf_norm = lf_power / lf_hf_sum * 100
        hf_norm = hf_power / lf_hf_sum * 100
    else:
        lf_norm = np.nan
        hf_norm = np.nan
    
    # LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    
    return FrequencyDomainHRV(
        vlf_power=vlf_power,
        lf_power=lf_power,
        hf_power=hf_power,
        total_power=total_power,
        lf_norm=lf_norm,
        hf_norm=hf_norm,
        lf_hf_ratio=lf_hf_ratio,
    )


def compute_hrv_metrics(
    peak_times: np.ndarray,
    segment_name: str,
    session_id: str,
    duration_sec: float,
    config: Config = default_config,
) -> HRVMetrics:
    """
    Compute complete HRV metrics from R-peak times.
    
    Parameters
    ----------
    peak_times : np.ndarray
        R-peak times in seconds.
    segment_name : str
        Segment identifier.
    session_id : str
        Session identifier.
    duration_sec : float
        Total segment duration in seconds.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    HRVMetrics
        Complete HRV analysis results.
    """
    quality_notes = []
    
    # Generate RR intervals
    if len(peak_times) < 2:
        quality_notes.append("✗ Insufficient peaks for HRV analysis")
        return HRVMetrics(
            segment_name=segment_name,
            session_id=session_id,
            duration_sec=duration_sec,
            n_beats=len(peak_times),
            n_valid_intervals=0,
            removal_rate=1.0,
            time_domain=TimeDomainHRV(
                mean_rr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan,
                mean_hr=np.nan, std_hr=np.nan, min_hr=np.nan, max_hr=np.nan,
            ),
            frequency_domain=FrequencyDomainHRV(
                vlf_power=np.nan, lf_power=np.nan, hf_power=np.nan,
                total_power=np.nan, lf_norm=np.nan, hf_norm=np.nan,
                lf_hf_ratio=np.nan,
            ),
            quality_notes=quality_notes,
        )
    
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(peak_times) * 1000
    
    # Clean RR intervals
    cleaning_result = clean_rr_intervals(rr_intervals, config)
    
    if cleaning_result.removal_rate > 0.2:
        quality_notes.append(
            f"⚠ High artifact rate ({cleaning_result.removal_rate:.1%} removed)"
        )
    else:
        quality_notes.append(
            f"✓ Artifact rate acceptable ({cleaning_result.removal_rate:.1%} removed)"
        )
    
    if cleaning_result.n_removed_physiological > 0:
        quality_notes.append(
            f"  - {cleaning_result.n_removed_physiological} outside physiological range"
        )
    if cleaning_result.n_removed_mad > 0:
        quality_notes.append(
            f"  - {cleaning_result.n_removed_mad} outliers (MAD method)"
        )
    
    rr_cleaned = cleaning_result.rr_cleaned
    
    # Minimum data requirements
    min_required = 30  # At least 30 intervals for reliable HRV
    if len(rr_cleaned) < min_required:
        quality_notes.append(
            f"⚠ Insufficient valid intervals ({len(rr_cleaned)} < {min_required})"
        )
    
    # Compute time-domain metrics
    time_domain = compute_time_domain(rr_cleaned)
    
    # Compute frequency-domain metrics
    frequency_domain = compute_frequency_domain(rr_cleaned, config)
    
    # Add HR summary to notes
    if not np.isnan(time_domain.mean_hr):
        quality_notes.append(
            f"✓ Mean HR: {time_domain.mean_hr:.1f} bpm "
            f"(range: {time_domain.min_hr:.0f}-{time_domain.max_hr:.0f})"
        )
    
    return HRVMetrics(
        segment_name=segment_name,
        session_id=session_id,
        duration_sec=duration_sec,
        n_beats=len(peak_times),
        n_valid_intervals=len(rr_cleaned),
        removal_rate=cleaning_result.removal_rate,
        time_domain=time_domain,
        frequency_domain=frequency_domain,
        quality_notes=quality_notes,
    )
