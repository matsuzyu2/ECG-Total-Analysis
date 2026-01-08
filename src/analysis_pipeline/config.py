"""analysis_pipeline configuration.

Centralizes configurable parameters for signal processing, R-peak detection,
and HRV analysis.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Pipeline configuration parameters."""
    
    # ==========================================================================
    # Signal Acquisition Parameters
    # ==========================================================================
    SAMPLING_RATE: int = 500  # Hz (Polar H10 / Cognionics default)
    
    # ==========================================================================
    # Signal Polarity Detection
    # ==========================================================================
    # Skewness threshold for automatic signal inversion
    # If skewness < SKEWNESS_THRESHOLD, the signal is considered inverted
    SKEWNESS_THRESHOLD: float = -0.8
    
    # ==========================================================================
    # Bandpass Filter Parameters (Zero-phase Butterworth)
    # ==========================================================================
    BANDPASS_LOW: float = 0.5   # Hz - removes baseline wander
    BANDPASS_HIGH: float = 40.0  # Hz - removes high-frequency noise/EMG
    FILTER_ORDER: int = 4        # Butterworth filter order
    
    # Edge artifact handling: padding duration in seconds
    # This padding is added before filtering and removed before R-peak detection
    # to prevent edge artifacts during bandpass filtering
    FILTER_PADDING_SEC: float = 15.0  # seconds
    
    @property
    def BANDPASS(self) -> Tuple[float, float]:
        """Return bandpass frequency range as tuple."""
        return (self.BANDPASS_LOW, self.BANDPASS_HIGH)
    
    # ==========================================================================
    # R-Peak Detection Parameters
    # ==========================================================================
    # Default detector matches `References/Analysis.R::peak_detect()`
    RPEAK_METHOD: str = "sgolay_derivative"
    
    # ==========================================================================
    # RR Interval Cleaning Parameters
    # ==========================================================================
    # Physiological range for valid RR intervals (in milliseconds)
    RR_MIN_MS: float = 300.0   # ~200 bpm max
    RR_MAX_MS: float = 2000.0  # ~30 bpm min
    
    # MAD (Median Absolute Deviation) multiplier for outlier detection
    MAD_THRESHOLD: float = 3.5
    
    # ==========================================================================
    # HRV Frequency Domain Parameters
    # ==========================================================================
    # Frequency bands for HRV analysis (Hz)
    VLF_BAND: Tuple[float, float] = (0.003, 0.04)
    LF_BAND: Tuple[float, float] = (0.04, 0.15)
    HF_BAND: Tuple[float, float] = (0.15, 0.4)
    
    # Welch PSD parameters for HRV
    HRV_NPERSEG: int = 256  # Segment length for Welch method
    
    # ==========================================================================
    # Directory Structure
    # ==========================================================================
    # Base paths (will be resolved relative to project root)
    DATA_DIR: str = "Data"
    RESULTS_DIR: str = "Results"
    
    # Subdirectories
    PROCESSED_SUBDIR: str = "Processed"
    SEGMENTS_SUBDIR: str = "split_segments"
    STATS_SUBDIR: str = "stats"
    
    # ==========================================================================
    # Output File Naming
    # ==========================================================================
    DIAGNOSIS_PREFIX: str = "diagnosis_"
    PEAKS_PREFIX: str = "peaks_"
    HRV_SUMMARY_FILE: str = "hrv_summary.csv"
    
    # ==========================================================================
    # Visualization Parameters
    # ==========================================================================
    # Number of seconds to show in zoomed waveform view
    WAVEFORM_ZOOM_SECONDS: float = 10.0
    
    # PSD frequency range for visualization
    PSD_FREQ_MAX: float = 100.0  # Hz
    
    def get_project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def get_data_dir(self) -> Path:
        """Get data directory path."""
        return self.get_project_root() / self.DATA_DIR
    
    def get_results_dir(self) -> Path:
        """Get results directory path."""
        return self.get_project_root() / self.RESULTS_DIR
    
    def get_session_results_dir(self, session_id: str) -> Path:
        """Get session-specific results directory."""
        results_dir = self.get_results_dir() / session_id
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def get_hrv_summaries_dir(self) -> Path:
        """Get centralized directory for HRV summary CSV outputs."""
        summaries_dir = self.get_results_dir() / "hrv_summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        return summaries_dir

    def get_stats_dir(self) -> Path:
        """Get centralized directory for group stats outputs."""
        stats_dir = self.get_results_dir() / self.STATS_SUBDIR
        stats_dir.mkdir(parents=True, exist_ok=True)
        return stats_dir
    
    def get_segments_dir(self, session_id: str) -> Path:
        """Get split_segments directory for a session."""
        return (
            self.get_data_dir() 
            / self.PROCESSED_SUBDIR 
            / session_id 
            / self.SEGMENTS_SUBDIR
        )
    
    def get_diagnosis_path(self, session_id: str, segment_name: str) -> Path:
        """Get path for diagnosis HTML report."""
        return (
            self.get_session_results_dir(session_id) 
            / f"{self.DIAGNOSIS_PREFIX}{segment_name}.html"
        )
    
    def get_peaks_path(self, session_id: str, segment_name: str) -> Path:
        """Get path for peaks JSON file."""
        return (
            self.get_session_results_dir(session_id) 
            / f"{self.PEAKS_PREFIX}{segment_name}.json"
        )
    
    def get_hrv_summary_path(self, session_id: str) -> Path:
        """Get path for HRV summary CSV."""
        return self.get_hrv_summaries_dir() / f"{session_id}_{self.HRV_SUMMARY_FILE}"


# Default configuration instance
default_config = Config()
