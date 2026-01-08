"""
Signal diagnosis and EDA (Exploratory Data Analysis) module.

Provides:
- Polarity detection via skewness analysis
- Automatic signal inversion
- Power Spectral Density (PSD) analysis
- Interactive diagnostic HTML reports with Plotly
"""

from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config, default_config


@dataclass
class PolarityResult:
    """Result of polarity check."""
    skewness: float
    is_inverted: bool
    recommendation: str


@dataclass
class PSDResult:
    """Result of PSD analysis."""
    frequencies: np.ndarray
    psd: np.ndarray
    dominant_freq: float
    power_50hz: float  # Power at 50Hz (European line noise)
    power_60hz: float  # Power at 60Hz (US/Japan line noise)
    has_line_noise: bool
    noise_notes: List[str]


def check_polarity(
    ecg: np.ndarray,
    config: Config = default_config,
) -> PolarityResult:
    """
    Check signal polarity using skewness analysis.
    
    ECG signals typically have positive skewness due to the sharp R-peak.
    Negative skewness suggests the signal is inverted.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal (should be filtered for best results).
    config : Config
        Pipeline configuration with SKEWNESS_THRESHOLD.
    
    Returns
    -------
    PolarityResult
        Polarity analysis result with recommendation.
    """
    skewness = stats.skew(ecg)
    is_inverted = skewness < config.SKEWNESS_THRESHOLD
    
    if is_inverted:
        recommendation = (
            f"Signal appears INVERTED (skewness={skewness:.3f} < {config.SKEWNESS_THRESHOLD}). "
            "Automatic inversion recommended."
        )
    else:
        recommendation = (
            f"Signal polarity OK (skewness={skewness:.3f}). "
            "No inversion needed."
        )
    
    return PolarityResult(
        skewness=skewness,
        is_inverted=is_inverted,
        recommendation=recommendation,
    )


def auto_invert(
    ecg: np.ndarray,
    polarity_result: PolarityResult,
) -> Tuple[np.ndarray, bool]:
    """
    Automatically invert ECG signal if needed based on polarity check.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    polarity_result : PolarityResult
        Result from check_polarity().
    
    Returns
    -------
    Tuple[np.ndarray, bool]
        (Corrected ECG signal, was_inverted flag)
    """
    if polarity_result.is_inverted:
        return -ecg, True
    return ecg, False


def compute_psd(
    ecg: np.ndarray,
    fs: int,
    config: Config = default_config,
) -> PSDResult:
    """
    Compute Power Spectral Density using Welch's method.
    
    Analyzes the frequency content to detect:
    - Line noise (50/60 Hz)
    - EMG contamination (high frequency power)
    - Low frequency drift
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    fs : int
        Sampling frequency in Hz.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    PSDResult
        PSD analysis result with noise indicators.
    """
    # Compute PSD using Welch's method
    nperseg = min(fs * 4, len(ecg))  # 4-second segments or less
    frequencies, psd = signal.welch(ecg, fs=fs, nperseg=nperseg)
    
    # Find dominant frequency (in physiological range)
    ecg_band_mask = (frequencies >= 0.5) & (frequencies <= 40)
    if np.any(ecg_band_mask):
        ecg_band_psd = psd.copy()
        ecg_band_psd[~ecg_band_mask] = 0
        dominant_freq = frequencies[np.argmax(ecg_band_psd)]
    else:
        dominant_freq = 0.0
    
    # Check for line noise
    noise_notes = []
    
    # 50 Hz power (European)
    idx_50 = np.argmin(np.abs(frequencies - 50))
    power_50hz = psd[idx_50] if idx_50 < len(psd) else 0.0
    
    # 60 Hz power (US/Japan)
    idx_60 = np.argmin(np.abs(frequencies - 60))
    power_60hz = psd[idx_60] if idx_60 < len(psd) else 0.0
    
    # Calculate median power for comparison
    median_power = np.median(psd[ecg_band_mask]) if np.any(ecg_band_mask) else 1.0
    
    # Detect significant line noise (10x median power)
    has_line_noise = False
    if power_50hz > median_power * 10:
        has_line_noise = True
        noise_notes.append(f"⚠ 50Hz line noise detected (power ratio: {power_50hz/median_power:.1f}x)")
    
    if power_60hz > median_power * 10:
        has_line_noise = True
        noise_notes.append(f"⚠ 60Hz line noise detected (power ratio: {power_60hz/median_power:.1f}x)")
    
    # Check for EMG contamination (high power above 40Hz)
    high_freq_mask = frequencies > 40
    if np.any(high_freq_mask):
        high_freq_power = np.mean(psd[high_freq_mask])
        if high_freq_power > median_power * 0.5:
            noise_notes.append(f"⚠ High-frequency noise/EMG detected (>40Hz)")
    
    # Check for baseline wander (high power below 0.5Hz)
    low_freq_mask = frequencies < 0.5
    if np.any(low_freq_mask):
        low_freq_power = np.mean(psd[low_freq_mask])
        if low_freq_power > median_power * 5:
            noise_notes.append(f"⚠ Baseline wander detected (<0.5Hz)")
    
    if not noise_notes:
        noise_notes.append("✓ No significant noise artifacts detected")
    
    return PSDResult(
        frequencies=frequencies,
        psd=psd,
        dominant_freq=dominant_freq,
        power_50hz=power_50hz,
        power_60hz=power_60hz,
        has_line_noise=has_line_noise,
        noise_notes=noise_notes,
    )


def create_diagnosis_report(
    ecg_raw: np.ndarray,
    ecg_filtered: np.ndarray,
    time: np.ndarray,
    peak_indices: np.ndarray,
    polarity_result: PolarityResult,
    psd_result: PSDResult,
    segment_name: str,
    session_id: str,
    was_converted: bool,
    was_inverted: bool,
    fs: int,
    config: Config = default_config,
) -> str:
    """
    Create interactive HTML diagnostic report using Plotly.
    
    The report contains:
    - Top row: Raw vs filtered waveform comparison
    - Middle row: Histogram (polarity) and PSD (noise)
    - Bottom row: Detected R-peaks overlay with zoom capability
    
    Parameters
    ----------
    ecg_raw : np.ndarray
        Raw ECG signal (after unit conversion). If automatic inversion was applied,
        pass the original (pre-inversion) signal here so the report can show the
        inverted polarity.
    ecg_filtered : np.ndarray
        Filtered ECG signal.
    time : np.ndarray
        Time array in seconds.
    peak_indices : np.ndarray
        Indices of detected R-peaks.
    polarity_result : PolarityResult
        Result of polarity analysis.
    psd_result : PSDResult
        Result of PSD analysis.
    segment_name : str
        Name of the segment.
    session_id : str
        Session identifier.
    was_converted : bool
        Whether µV→mV conversion was applied.
    was_inverted : bool
        Whether signal was inverted.
    fs : int
        Sampling frequency.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    str
        HTML string of the diagnostic report.
    """
    # Create 3x2 subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Raw vs Filtered ECG (Full)",
            "Raw vs Filtered ECG (Zoomed 10s)",
            "Signal Histogram (Polarity Check)",
            "Power Spectral Density (Noise Check)",
            "Detected R-Peaks (Full)",
            "Detected R-Peaks (Zoomed 10s)",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    
    # Color scheme
    color_raw = "rgba(150, 150, 150, 0.5)"
    color_filtered = "rgb(0, 100, 200)"
    color_peaks = "rgb(255, 0, 0)"
    
    # Calculate zoom window (first 10 seconds)
    zoom_samples = int(config.WAVEFORM_ZOOM_SECONDS * fs)
    zoom_end = min(zoom_samples, len(time))
    
    # Row 1: Raw vs Filtered comparison
    # Full view
    fig.add_trace(
        go.Scatter(
            x=time, y=ecg_raw,
            mode='lines', name='Raw',
            line=dict(color=color_raw, width=1),
            legendgroup="raw",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=ecg_filtered,
            mode='lines', name='Filtered',
            line=dict(color=color_filtered, width=1),
            legendgroup="filtered",
        ),
        row=1, col=1
    )
    
    # Zoomed view
    fig.add_trace(
        go.Scatter(
            x=time[:zoom_end], y=ecg_raw[:zoom_end],
            mode='lines', name='Raw',
            line=dict(color=color_raw, width=1),
            legendgroup="raw",
            showlegend=False,
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=time[:zoom_end], y=ecg_filtered[:zoom_end],
            mode='lines', name='Filtered',
            line=dict(color=color_filtered, width=1),
            legendgroup="filtered",
            showlegend=False,
        ),
        row=1, col=2
    )
    
    # Row 2: Histogram and PSD
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=ecg_filtered,
            nbinsx=100,
            name='ECG Distribution',
            marker_color=color_filtered,
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # Add skewness annotation
    skew_text = f"Skewness: {polarity_result.skewness:.3f}"
    if polarity_result.is_inverted and was_inverted:
        skew_text += " (INVERTED → Corrected)"
    elif polarity_result.is_inverted and not was_inverted:
        skew_text += " (INVERTED; not corrected)"
    
    # PSD plot
    fig.add_trace(
        go.Scatter(
            x=psd_result.frequencies,
            y=10 * np.log10(psd_result.psd + 1e-12),  # Convert to dB
            mode='lines',
            name='PSD',
            line=dict(color=color_filtered, width=1.5),
            showlegend=False,
        ),
        row=2, col=2
    )
    
    # Add vertical lines for filter cutoffs
    fig.add_vline(x=config.BANDPASS_LOW, line_dash="dash", line_color="green", 
                  annotation_text=f"{config.BANDPASS_LOW}Hz", row=2, col=2)
    fig.add_vline(x=config.BANDPASS_HIGH, line_dash="dash", line_color="green",
                  annotation_text=f"{config.BANDPASS_HIGH}Hz", row=2, col=2)
    
    # Add vertical lines for line noise frequencies
    fig.add_vline(x=50, line_dash="dot", line_color="orange",
                  annotation_text="50Hz", row=2, col=2)
    fig.add_vline(x=60, line_dash="dot", line_color="red",
                  annotation_text="60Hz", row=2, col=2)
    
    # Row 3: R-peak overlay
    # Full view
    fig.add_trace(
        go.Scatter(
            x=time, y=ecg_filtered,
            mode='lines', name='ECG',
            line=dict(color=color_filtered, width=1),
            legendgroup="ecg",
        ),
        row=3, col=1
    )
    
    if len(peak_indices) > 0:
        peak_times = time[peak_indices]
        peak_values = ecg_filtered[peak_indices]
        fig.add_trace(
            go.Scatter(
                x=peak_times, y=peak_values,
                mode='markers', name='R-peaks',
                marker=dict(color=color_peaks, size=6, symbol='x'),
                legendgroup="peaks",
            ),
            row=3, col=1
        )
    
    # Zoomed view
    fig.add_trace(
        go.Scatter(
            x=time[:zoom_end], y=ecg_filtered[:zoom_end],
            mode='lines', name='ECG',
            line=dict(color=color_filtered, width=1),
            legendgroup="ecg",
            showlegend=False,
        ),
        row=3, col=2
    )
    
    # Add peaks in zoomed view
    zoom_peak_mask = peak_indices < zoom_end
    if np.any(zoom_peak_mask):
        zoom_peaks = peak_indices[zoom_peak_mask]
        fig.add_trace(
            go.Scatter(
                x=time[zoom_peaks], y=ecg_filtered[zoom_peaks],
                mode='markers', name='R-peaks',
                marker=dict(color=color_peaks, size=8, symbol='x'),
                legendgroup="peaks",
                showlegend=False,
            ),
            row=3, col=2
        )
    
    # Calculate statistics
    n_peaks = len(peak_indices)
    duration = time[-1] - time[0] if len(time) > 0 else 0
    avg_hr = (n_peaks / duration * 60) if duration > 0 else 0
    
    # Build summary text
    summary_lines = [
        f"<b>Session:</b> {session_id} | <b>Segment:</b> {segment_name}",
        f"<b>Duration:</b> {duration:.1f}s | <b>Samples:</b> {len(ecg_filtered):,} | <b>Fs:</b> {fs}Hz",
        f"<b>Unit Conversion:</b> {'µV→mV applied' if was_converted else 'Not needed (already mV)'}",
        f"<b>Polarity:</b> {skew_text}",
        f"<b>R-peaks detected:</b> {n_peaks} | <b>Avg HR:</b> {avg_hr:.1f} bpm",
        "<b>Noise Analysis:</b> " + " | ".join(psd_result.noise_notes),
    ]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<br>".join(summary_lines),
            x=0.5,
            xanchor='center',
            font=dict(size=12),
        ),
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Amplitude (mV)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude (mV)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=3, col=2)
    
    # Limit PSD x-axis
    fig.update_xaxes(range=[0, config.PSD_FREQ_MAX], row=2, col=2)
    
    # Generate HTML
    html_content = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        }
    )
    
    return html_content
