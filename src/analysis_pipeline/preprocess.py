"""
ECG signal preprocessing module.

Provides zero-phase Butterworth bandpass filtering
to remove baseline wander and high-frequency noise.
"""

from typing import Tuple, Optional

import numpy as np
from scipy import signal

from .config import Config, default_config


def filter_ecg(
    ecg: np.ndarray,
    fs: int = None,
    lowcut: float = None,
    highcut: float = None,
    order: int = None,
    config: Config = default_config,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to ECG signal.
    
    Uses scipy.signal.filtfilt for zero-phase filtering,
    which prevents phase distortion that could affect R-peak timing.
    
    Parameters
    ----------
    ecg : np.ndarray
        Raw ECG signal.
    fs : int, optional
        Sampling frequency in Hz. Defaults to config.SAMPLING_RATE.
    lowcut : float, optional
        Low cutoff frequency in Hz. Defaults to config.BANDPASS_LOW.
    highcut : float, optional
        High cutoff frequency in Hz. Defaults to config.BANDPASS_HIGH.
    order : int, optional
        Filter order. Defaults to config.FILTER_ORDER.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    
    Notes
    -----
    - Low cutoff (0.5 Hz): Removes baseline wander from respiration/movement
    - High cutoff (40 Hz): Removes EMG noise while preserving QRS complex
    - Zero-phase filtering: Applies filter forward and backward to eliminate
      phase delay, critical for accurate R-peak timing
    """
    # Use config defaults if not specified
    if fs is None:
        fs = config.SAMPLING_RATE
    if lowcut is None:
        lowcut = config.BANDPASS_LOW
    if highcut is None:
        highcut = config.BANDPASS_HIGH
    if order is None:
        order = config.FILTER_ORDER
    
    # Calculate Nyquist frequency
    nyquist = fs / 2.0
    
    # Normalize frequencies
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure valid frequency range
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.999
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering
    # Use padlen to handle edge effects
    padlen = min(3 * max(len(a), len(b)), len(ecg) - 1)
    
    try:
        ecg_filtered = signal.filtfilt(b, a, ecg, padlen=padlen)
    except ValueError as e:
        # Fallback for very short signals
        print(f"  ⚠ Filter warning: {e}. Using minimum padding.")
        ecg_filtered = signal.filtfilt(b, a, ecg, padlen=min(10, len(ecg) - 1))
    
    return ecg_filtered


def highpass_filter(
    ecg: np.ndarray,
    fs: int = None,
    cutoff: float = 0.5,
    order: int = None,
    config: Config = default_config,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth highpass filter.
    
    Useful for removing only baseline wander without
    affecting high-frequency content.
    
    Parameters
    ----------
    ecg : np.ndarray
        Raw ECG signal.
    fs : int, optional
        Sampling frequency in Hz.
    cutoff : float
        Cutoff frequency in Hz.
    order : int, optional
        Filter order.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    np.ndarray
        Highpass filtered ECG signal.
    """
    if fs is None:
        fs = config.SAMPLING_RATE
    if order is None:
        order = config.FILTER_ORDER
    
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff <= 0:
        normalized_cutoff = 0.001
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    
    padlen = min(3 * max(len(a), len(b)), len(ecg) - 1)
    
    try:
        return signal.filtfilt(b, a, ecg, padlen=padlen)
    except ValueError:
        return signal.filtfilt(b, a, ecg, padlen=min(10, len(ecg) - 1))


def lowpass_filter(
    ecg: np.ndarray,
    fs: int = None,
    cutoff: float = 40.0,
    order: int = None,
    config: Config = default_config,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth lowpass filter.
    
    Useful for removing high-frequency noise/EMG.
    
    Parameters
    ----------
    ecg : np.ndarray
        Raw ECG signal.
    fs : int, optional
        Sampling frequency in Hz.
    cutoff : float
        Cutoff frequency in Hz.
    order : int, optional
        Filter order.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    np.ndarray
        Lowpass filtered ECG signal.
    """
    if fs is None:
        fs = config.SAMPLING_RATE
    if order is None:
        order = config.FILTER_ORDER
    
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.999
    
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    padlen = min(3 * max(len(a), len(b)), len(ecg) - 1)
    
    try:
        return signal.filtfilt(b, a, ecg, padlen=padlen)
    except ValueError:
        return signal.filtfilt(b, a, ecg, padlen=min(10, len(ecg) - 1))


def notch_filter(
    ecg: np.ndarray,
    fs: int = None,
    freq: float = 50.0,
    quality: float = 30.0,
    config: Config = default_config,
) -> np.ndarray:
    """
    Apply notch filter to remove line noise.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    fs : int, optional
        Sampling frequency in Hz.
    freq : float
        Notch frequency in Hz (50 or 60 for line noise).
    quality : float
        Quality factor. Higher = narrower notch.
    config : Config
        Pipeline configuration.
    
    Returns
    -------
    np.ndarray
        Notch filtered ECG signal.
    """
    if fs is None:
        fs = config.SAMPLING_RATE
    
    # Design notch filter
    b, a = signal.iirnotch(freq, quality, fs)
    
    # Apply zero-phase filtering
    return signal.filtfilt(b, a, ecg)


def detrend_signal(ecg: np.ndarray, type: str = 'linear') -> np.ndarray:
    """
    Remove linear or constant trend from signal.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    type : str
        'linear' or 'constant'.
    
    Returns
    -------
    np.ndarray
        Detrended signal.
    """
    from scipy.signal import detrend
    return detrend(ecg, type=type)
