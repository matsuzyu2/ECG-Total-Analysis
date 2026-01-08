"""AR-based HRV frequency analysis for short windows.

This module adds an alternative to Welch PSD for HRV frequency-domain metrics.
It is designed for ~60s windows where FFT/Welch can be unstable.

Approach:
- RR intervals (ms) -> cubic/linear interpolation to evenly sampled series (4 Hz)
- Linear detrend (scipy.signal.detrend)
- AR PSD estimation via Burg method (spectrum.pburg)
- Integrate LF/HF band powers and compute LF/HF ratio

Outputs:
- Frequency-domain metrics (VLF/LF/HF/total, LFnu/HFnu, LF/HF)
- PSD arrays for plotting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import interpolate, signal

from .config import Config, default_config


@dataclass
class HRVPSDResult:
    """Container for HRV PSD + derived band metrics."""

    freqs_hz: np.ndarray
    psd: np.ndarray
    fs_resample_hz: float
    method: str
    ar_order: int
    nfft: int

    vlf_power: float
    lf_power: float
    hf_power: float
    total_power: float
    lf_norm: float
    hf_norm: float
    lf_hf_ratio: float

    notes: Tuple[str, ...] = ()


def _integrate_band(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(mask):
        return 0.0
    try:
        return float(np.trapezoid(psd[mask], freqs[mask]))
    except AttributeError:
        return float(np.trapz(psd[mask], freqs[mask]))


def _rr_to_evenly_sampled(
    rr_intervals_ms: np.ndarray,
    *,
    fs_resample_hz: float,
) -> Optional[np.ndarray]:
    """Convert irregular RR (ms) to an evenly sampled series (ms) at fs_resample_hz."""
    if rr_intervals_ms.size < 10:
        return None

    rr_seconds = rr_intervals_ms / 1000.0
    t_rr = np.cumsum(rr_seconds)
    t_rr = np.insert(t_rr, 0, 0.0)[:-1]  # time at each RR sample

    if t_rr.size < 2:
        return None

    t_interp = np.arange(float(t_rr[0]), float(t_rr[-1]), 1.0 / fs_resample_hz)
    if t_interp.size < 10:
        return None

    # Prefer cubic, fallback to linear
    try:
        interp_func = interpolate.interp1d(t_rr, rr_intervals_ms, kind="cubic", fill_value="extrapolate")
        rr_interp = interp_func(t_interp)
    except Exception:
        interp_func = interpolate.interp1d(t_rr, rr_intervals_ms, kind="linear", fill_value="extrapolate")
        rr_interp = interp_func(t_interp)

    return np.asarray(rr_interp, dtype=np.float64)


def compute_hrv_psd_ar(
    rr_intervals_ms: np.ndarray,
    *,
    config: Config = default_config,
    fs_resample_hz: float = 4.0,
    ar_order: int = 16,
    nfft: int = 1024,
) -> HRVPSDResult:
    """Compute HRV PSD via AR(Burg) and derive LF/HF metrics.

    Notes
    -----
    - Designed for short segments (~60s). AR is generally more stable than Welch.
    - Uses linear detrending (scipy.signal.detrend) as recommended to suppress drift.
    """
    notes = []

    rr_even = _rr_to_evenly_sampled(rr_intervals_ms, fs_resample_hz=fs_resample_hz)
    if rr_even is None:
        raise ValueError("Insufficient RR data for frequency analysis (need >= ~10 samples after interpolation)")

    # Linear detrend (explicit requirement)
    rr_detrended = signal.detrend(rr_even, type="linear")

    try:
        from spectrum import pburg
    except Exception as e:
        raise ImportError(
            "AR PSD requires the 'spectrum' package. Install it with: pip install spectrum"
        ) from e

    # Burg PSD
    # spectrum.pburg returns an object with `.psd` and `.frequencies()`.
    burg = pburg(rr_detrended, order=int(ar_order), NFFT=int(nfft), sampling=float(fs_resample_hz))
    freqs = np.asarray(burg.frequencies(), dtype=np.float64)
    psd = np.asarray(burg.psd, dtype=np.float64)

    if freqs.size != psd.size or freqs.size == 0:
        raise ValueError("AR PSD computation returned empty arrays")

    # Restrict to HRV-relevant range (0-0.5 Hz typically)
    keep = (freqs >= 0.0) & (freqs <= 0.5)
    if np.any(keep):
        freqs = freqs[keep]
        psd = psd[keep]
    else:
        notes.append("⚠ PSD has no points in 0-0.5 Hz range")

    vlf_power = _integrate_band(freqs, psd, config.VLF_BAND)
    lf_power = _integrate_band(freqs, psd, config.LF_BAND)
    hf_power = _integrate_band(freqs, psd, config.HF_BAND)
    total_power = vlf_power + lf_power + hf_power

    lf_hf_sum = lf_power + hf_power
    if lf_hf_sum > 0:
        lf_norm = lf_power / lf_hf_sum * 100.0
        hf_norm = hf_power / lf_hf_sum * 100.0
    else:
        lf_norm = float("nan")
        hf_norm = float("nan")
        notes.append("⚠ LF+HF power is zero; LFnu/HFnu undefined")

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else float("nan")
    if not np.isfinite(lf_hf_ratio):
        notes.append("⚠ HF power is zero/invalid; LF/HF undefined")

    return HRVPSDResult(
        freqs_hz=freqs,
        psd=psd,
        fs_resample_hz=float(fs_resample_hz),
        method="AR(Burg)",
        ar_order=int(ar_order),
        nfft=int(nfft),
        vlf_power=float(vlf_power),
        lf_power=float(lf_power),
        hf_power=float(hf_power),
        total_power=float(total_power),
        lf_norm=float(lf_norm),
        hf_norm=float(hf_norm),
        lf_hf_ratio=float(lf_hf_ratio),
        notes=tuple(notes),
    )


def plot_hrv_psd_with_lfhf(
    psd_result: HRVPSDResult,
    *,
    title: str,
    output_html,
    config: Config = default_config,
) -> None:
    """Plot PSD with LF/HF bands and LF/HF annotation (legacy helper).

    Prefer using `hrv_psd_html_fragment` + `write_hrv_psd_summary_html` to bundle
    multiple segments into one HTML.
    """
    import pathlib
    fig = make_hrv_psd_figure(psd_result, title=title, config=config)
    output_html = str(output_html)
    pathlib.Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)


def make_hrv_psd_figure(
    psd_result: HRVPSDResult,
    *,
    title: str,
    config: Config = default_config,
):
    """Create a Plotly Figure for HRV PSD with LF/HF annotation."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=psd_result.freqs_hz,
            y=psd_result.psd,
            mode="lines",
            name="PSD",
        )
    )

    # Band shading
    def add_band(band: Tuple[float, float], label: str, fill_rgba: str) -> None:
        fig.add_vrect(
            x0=float(band[0]),
            x1=float(band[1]),
            fillcolor=fill_rgba,
            opacity=0.18,
            line_width=0,
            layer="below",
            annotation_text=label,
            annotation_position="top left",
        )

    add_band(config.LF_BAND, "LF", "#1f77b4")
    add_band(config.HF_BAND, "HF", "#d62728")

    txt = (
        f"{psd_result.method} (order={psd_result.ar_order})<br>"
        f"LF={psd_result.lf_power:.2f} ms², HF={psd_result.hf_power:.2f} ms²<br>"
        f"LF/HF={psd_result.lf_hf_ratio:.2f}"
    )
    fig.add_annotation(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=txt,
        showarrow=False,
        align="right",
        bordercolor="#ccc",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.85)",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (a.u.)",
        hovermode="x unified",
    )
    return fig


def hrv_psd_html_fragment(
    psd_result: HRVPSDResult,
    *,
    title: str,
    config: Config = default_config,
    include_plotlyjs,
) -> str:
    """Return an HTML fragment for a single PSD figure.

    Parameters
    ----------
    include_plotlyjs:
        Pass 'cdn' for the first plot, and False for subsequent plots.
    """
    import plotly.io as pio

    fig = make_hrv_psd_figure(psd_result, title=title, config=config)
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs)


def write_hrv_psd_summary_html(
    *,
    session_id: str,
    items: list[tuple[str, str]],
    output_html,
) -> None:
    """Write a single HTML that contains multiple PSD plots.

    items:
        List of (segment_name, html_fragment)
    """
    import pathlib

    out = pathlib.Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    for segment_name, frag in items:
        blocks.append(
            "\n".join(
                [
                    '<div class="card">',
                    f'  <div class="title">{segment_name}</div>',
                    '  <div class="plot">',
                    frag,
                    "  </div>",
                    "</div>",
                ]
            )
        )

    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8" />',
            '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
            f"  <title>HRV PSD (AR/Burg) - {session_id}</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 16px; }",
            "    h1 { margin: 0 0 12px 0; font-size: 20px; }",
            "    .grid { display: flex; flex-wrap: wrap; gap: 12px; }",
            "    .card { flex: 1 1 560px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }",
            "    .title { padding: 8px 10px; font-weight: 600; border-bottom: 1px solid #eee; background: #fafafa; }",
            "    .plot { padding: 8px 10px; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>HRV PSD (AR/Burg) - {session_id}</h1>",
            '  <div class="grid">',
            *blocks,
            "  </div>",
            "</body>",
            "</html>",
        ]
    )

    out.write_text(html, encoding="utf-8")
