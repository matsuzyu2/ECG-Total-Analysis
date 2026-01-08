# Edge Artifact Handling Documentation

## Overview

This implementation adds 15-second padding buffers to ECG segment boundaries to prevent edge artifacts during bandpass filtering.

## Problem

When ECG data is sliced exactly at trigger timestamps and then filtered with a bandpass filter, edge artifacts occur at the segment boundaries. These artifacts can distort the signal and affect R-peak detection accuracy near the edges.

## Solution

The solution implements a three-step process:

1. **Add Padding**: When slicing raw ECG data based on triggers, add a 15-second buffer to both the start and end of the segment
2. **Apply Filter**: Apply the bandpass filter to the buffered segment
3. **Trim Buffer**: After filtering but before R-peak detection, remove the 15-second buffer from both ends

## Implementation Details

### Configuration

The padding duration is configurable in `src/analysis_pipeline/config.py`:

```python
FILTER_PADDING_SEC: float = 15.0  # seconds
```

### Data Slicing (split_by_annotation.py)

When creating segment CSV files, the script:
- Calculates timestamp offsets for the padding duration
- Extends the time range by ±15 seconds
- Adds metadata columns `_original_start_ts` and `_original_end_ts` to track the original boundaries

### Data Loading (io_utils.py)

The `SegmentData` dataclass includes padding metadata:
- `has_padding`: boolean flag indicating if padding is present
- `original_start_idx`: sample index where the original segment starts
- `original_end_idx`: sample index where the original segment ends

The `load_segment_csv()` function:
- Detects padding metadata columns
- Calculates the original segment boundaries
- Stores boundary indices for later use

### Signal Processing (run_signal_diagnosis.py)

The processing pipeline:
1. Loads the segment (with padding if present)
2. Applies bandpass filtering to the full padded signal
3. Trims the padding using the stored boundary indices
4. Performs R-peak detection on the trimmed signal

## Benefits

- **Eliminates Edge Artifacts**: Padding ensures the filter has sufficient context at boundaries
- **Maintains Accuracy**: R-peaks are only detected on the original segment duration
- **Backward Compatible**: Works with both padded and non-padded segments
- **Configurable**: Padding duration can be adjusted in the configuration

## Testing

The implementation includes comprehensive tests:

1. **Unit Tests** (`/tmp/test_padding.py`):
   - Configuration parameter validation
   - Data structure verification
   - Timestamp offset calculation

2. **Integration Test** (`/tmp/test_integration.py`):
   - Full pipeline workflow
   - Edge quality verification
   - Boundary correctness validation

## Usage

### For New Data Processing

1. Run `split_by_annotation.py` to create segment CSVs with padding
2. Run `run_signal_diagnosis.py` as usual - it will automatically detect and handle padding

### For Existing Data

Existing segment CSVs without padding metadata will continue to work as before. The pipeline automatically detects whether padding is present.

## Technical Notes

### Timestamp Handling

The implementation handles different timestamp formats:
- Numeric timestamps (seconds since epoch)
- DateTime timestamps
- Falls back to sample-based estimation if conversion fails

### Array Indexing

Python slicing syntax `[start:end]` is exclusive at the end, so:
- `[original_start_idx:original_end_idx+1]` includes both boundary samples
- Bounds checking prevents IndexError if indices are invalid

### Sampling Rate

The implementation uses the configurable sampling rate from `config.py` to ensure consistency across all modules.

## Example

For a 10-second segment:
- Original segment: 5000 samples at 500 Hz
- With padding: 20000 samples (15s + 10s + 15s)
- After filtering and trimming: 5000 samples (original duration)

The filtered signal at the edges is now clean because the filter had 15 seconds of context on each side.
