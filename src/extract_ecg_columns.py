from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


DESIRED_COLUMNS = [
    "Time (s)",
    "ExGa 1(uV)",
    "Packet Counter(DIGITAL)",
    "TRIGGER(DIGITAL)",
]

TIMESTAMP_METADATA_KEY = "Recording Time Stamp:"
TIMESTAMP_COLUMN_NAME = "Timestamp"


def _normalize_column_name(name: str) -> str:
    # Collapse whitespace and normalize trimming so we can match robustly.
    return re.sub(r"\s+", " ", str(name).strip())


def _read_metadata_lines(input_path: Path, skiprows: int) -> list[str]:
    if skiprows <= 0:
        return []

    metadata: list[str] = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as src:
        for _ in range(skiprows):
            line = src.readline()
            if not line:
                break
            metadata.append(line.rstrip("\r\n"))

    return metadata


def _write_metadata_header(output_path: Path, metadata_lines: list[str]) -> bool:
    if not metadata_lines:
        return False

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        for line in metadata_lines:
            dst.write(line + "\n")
        dst.write("\n")  # Separate metadata from the CSV header row.

    return True


def _parse_recording_start(metadata_lines: list[str]) -> datetime | None:
    for line in metadata_lines:
        if line.startswith(TIMESTAMP_METADATA_KEY):
            raw_value = line[len(TIMESTAMP_METADATA_KEY) :].strip()
            try:
                return datetime.fromisoformat(raw_value)
            except ValueError:
                print(f"⚠️  Unable to parse recording timestamp: {raw_value}")
                return None
    return None


def _resolve_columns(input_path: Path, delimiter: str, skiprows: int, thousands: str | None) -> list[str]:
    header_df = pd.read_csv(
        input_path,
        delimiter=delimiter,
        skiprows=skiprows,
        nrows=0,
        thousands=thousands,
    )

    available = list(header_df.columns)
    normalized_to_actual: dict[str, str] = {
        _normalize_column_name(col): str(col) for col in available
    }

    resolved: list[str] = []
    missing: list[str] = []

    for wanted in DESIRED_COLUMNS:
        key = _normalize_column_name(wanted)
        if key in normalized_to_actual:
            resolved.append(normalized_to_actual[key])
        else:
            missing.append(wanted)

    if missing:
        available_preview = ", ".join([_normalize_column_name(c) for c in available[:30]])
        raise SystemExit(
            "Required columns not found: "
            + ", ".join(missing)
            + "\nAvailable columns (first 30): "
            + available_preview
        )

    return resolved


def extract_columns(
    *,
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    delimiter: str,
    skiprows: int,
    thousands: str | None,
) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    append_to_existing = output_path.exists() and output_path.stat().st_size > 0
    if append_to_existing:
        print(f"↪ Appending to existing file: {output_path}")
    else:
        print(f"Writing new file: {output_path}")

    metadata_lines = _read_metadata_lines(input_path=input_path, skiprows=skiprows)
    recording_start = _parse_recording_start(metadata_lines)

    resolved_columns = _resolve_columns(
        input_path=input_path,
        delimiter=delimiter,
        skiprows=skiprows,
        thousands=thousands,
    )

    if append_to_existing:
        # Do not rewrite metadata/header when appending.
        metadata_written = False
        first_chunk_mode = "a"
    else:
        metadata_written = _write_metadata_header(
            output_path=output_path,
            metadata_lines=metadata_lines,
        )
        first_chunk_mode = "a" if metadata_written else "w"

    reader = pd.read_csv(
        input_path,
        delimiter=delimiter,
        skiprows=skiprows,
        usecols=resolved_columns,
        chunksize=chunk_size,
        thousands=thousands,
    )

    time_column_name = next(
        (col for col in resolved_columns if _normalize_column_name(col) == _normalize_column_name("Time (s)")),
        None,
    )
    timestamp_origin = pd.Timestamp(recording_start) if recording_start else None

    # If we are appending, assume the destination already has a header.
    wrote_header = append_to_existing
    total_rows = 0
    warned_fractional_int_columns: set[str] = set()

    for chunk_index, chunk in enumerate(reader, start=1):
        # Minimal processing:
        # - Ensure numeric parsing for the 4 requested columns.
        # - TRIGGER is kept as-is (no event logic), just extracted.
        for col in chunk.columns:
            normalized = _normalize_column_name(col)

            if normalized in {"Packet Counter(DIGITAL)", "TRIGGER(DIGITAL)"}:
                numeric = pd.to_numeric(chunk[col], errors="coerce")
                fractional_mask = numeric.notna() & (numeric % 1 != 0)

                if fractional_mask.any():
                    if normalized not in warned_fractional_int_columns:
                        warned_fractional_int_columns.add(normalized)
                        print(
                            f"⚠️  Column '{col}' contains non-integer values; keeping as float."
                        )
                    chunk[col] = numeric
                else:
                    chunk[col] = numeric.astype("Int64")
            else:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        if timestamp_origin is not None and time_column_name in chunk.columns:
            time_seconds = pd.to_numeric(chunk[time_column_name], errors="coerce").fillna(0)
            offsets = pd.to_timedelta(time_seconds, unit="s")
            timestamps = timestamp_origin + offsets
            insert_at = chunk.columns.get_loc(time_column_name) + 1
            chunk.insert(insert_at, TIMESTAMP_COLUMN_NAME, timestamps)

        mode = first_chunk_mode if not wrote_header else "a"
        chunk.to_csv(output_path, mode=mode, header=not wrote_header, index=False)
        wrote_header = True

        total_rows += len(chunk)
        if chunk_index % 10 == 0:
            print(f"... extracted {total_rows:,} rows")

    print(f"✓ Done. Wrote {total_rows:,} rows to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ECG analysis columns from a Cognionics tab-delimited text file. "
            "TRIGGER(DIGITAL) is extracted but not further processed."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input .txt file path to analyze",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per chunk (default: 200000)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\t",
        help="Delimiter (default: tab)",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=4,
        help="Rows to skip before the header row (default: 4)",
    )
    parser.add_argument(
        "--thousands",
        type=str,
        default=",",
        help="Thousands separator for numeric fields (default: ',')",
    )

    args = parser.parse_args()

    # Generate output path automatically based on input file
    input_path = args.input_file
    # Remove patterns like "_01", "_02", etc. from the filename
    cleaned_stem = re.sub(r'_\d+$', '', input_path.stem)
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else input_path.parent
    output_path = output_dir / f"{cleaned_stem}_ext.csv"

    extract_columns(
        input_path=input_path,
        output_path=output_path,
        chunk_size=args.chunk_size,
        delimiter=args.delimiter,
        skiprows=args.skiprows,
        thousands=args.thousands if args.thousands else None,
    )


if __name__ == "__main__":
    main()
