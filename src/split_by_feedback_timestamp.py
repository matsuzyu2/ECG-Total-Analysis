#!/usr/bin/env python3
"""Split ECG by Feedback timestamps.

This script:
- Reads per-subject HHMM timestamps from Timestamp.csv
- Extracts 4 resting-HR windows (Control pre/post, Target pre/post)
- Adds ±buffer padding for filter edge artifacts
- Writes padded segments to Data/Processed/{session}/split_segments

Segment files include metadata columns:
  - _original_start_ts
  - _original_end_ts
which allow the downstream pipeline to trim padding after bandpass filtering.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SessionIdentity:
    session_id: str
    date_str: str  # YYYY_MM_DD
    subject_token: str  # e.g. TK, MS
    subject_id: str  # e.g. TK, MS_1
    bf_type: str  # Inc/Dec


FEEDBACK_COLUMNS_ORDER = [
    "Feedback_Con_pre",
    "Feedback_Con_post",
    "Feedback_pre",
    "Feedback_post",
]


def _detect_csv_header_skiprows(csv_path: Path) -> int:
    """Detect how many metadata lines to skip before the real header.

    `extract_ecg_columns.py` may write metadata lines followed by a blank line,
    then the real CSV header (with columns like 'Time (s)', 'Timestamp', ...).
    """
    max_probe = 50
    with csv_path.open("r", encoding="utf-8", errors="ignore") as src:
        for i in range(max_probe):
            line = src.readline()
            if not line:
                break
            # Heuristic: the real header line should include Time (s) and at least
            # one known ECG column name.
            if "Time (s)" in line and ("ExGa" in line or "TRIGGER" in line or "Packet" in line or "Timestamp" in line):
                return i
    return 0


def _find_timestamp_csv(data_dir: Path) -> Path:
    candidates = [
        data_dir / "Timestamp" / "Timestamp_2.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Timestamp.csv not found. Looked in: {', '.join(str(p) for p in candidates)}")


def _parse_session_id(session_id: str) -> Tuple[str, str]:
    m = re.match(r"^(\d{6})_(.+)$", session_id)
    if not m:
        raise ValueError(f"Invalid session id format: {session_id} (expected like 251216_TK)")
    yymmdd = m.group(1)
    token = m.group(2)
    return yymmdd, token


def _yymmdd_to_datestr(yymmdd: str) -> str:
    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    yyyy = 2000 + yy
    return f"{yyyy:04d}_{mm:02d}_{dd:02d}"


def _clean_hhmm_cell(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "nat"}:
        return None
    # Timestamp.csv sometimes contains triple quotes like """1438""".
    text = re.sub(r"[\"']", "", text).strip()
    # After stripping quotes, it should be digits.
    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def _hhmm_to_datetime(date_str: str, hhmm: int) -> datetime:
    # date_str: YYYY_MM_DD
    yyyy, mm, dd = (int(x) for x in date_str.split("_"))
    hour = hhmm // 100
    minute = hhmm % 100
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HHMM value: {hhmm}")
    return datetime(yyyy, mm, dd, hour, minute, 0)


def resolve_session_identity(*, session_id: str, data_dir: Path) -> SessionIdentity:
    subject_df = pd.read_csv(data_dir / "Subject" / "SubjectData.csv")
    yymmdd, subject_token = _parse_session_id(session_id)
    date_str = _yymmdd_to_datestr(yymmdd)

    matches = subject_df[subject_df["Date"] == date_str]
    if matches.empty:
        raise ValueError(f"No SubjectData row for Date={date_str} (session={session_id})")

    # Prefer exact match, else prefix match (handles MS vs MS_1)
    exact = matches[matches["Subject"] == subject_token]
    if len(exact) == 1:
        row = exact.iloc[0]
    else:
        pref = matches[matches["Subject"].astype(str).str.startswith(subject_token)]
        if len(pref) != 1:
            raise ValueError(
                f"Unable to uniquely resolve Subject for session={session_id}. "
                f"Candidates: {pref['Subject'].tolist()} (Date={date_str}, token={subject_token})"
            )
        row = pref.iloc[0]

    subject_id = str(row["Subject"])
    bf_type = str(row.get("BF_Type", ""))
    return SessionIdentity(
        session_id=session_id,
        date_str=date_str,
        subject_token=subject_token,
        subject_id=subject_id,
        bf_type=bf_type,
    )


def _read_feedback_hhmm_values(*, session: SessionIdentity, data_dir: Path) -> Dict[str, int]:
    ts_path = _find_timestamp_csv(data_dir)
    ts_df = pd.read_csv(ts_path)

    row_df = ts_df[(ts_df["Date"] == session.date_str) & (ts_df["Subject"] == session.subject_id)]
    if row_df.empty:
        raise ValueError(
            f"No Timestamp.csv row for Date={session.date_str}, Subject={session.subject_id} (session={session.session_id})."
        )
    row = row_df.iloc[0]

    values: Dict[str, int] = {}
    for col in ts_df.columns:
        if "Feedback" not in str(col):
            continue
        hhmm = _clean_hhmm_cell(row[col])
        if hhmm is None:
            continue
        values[str(col)] = hhmm

    # Enforce required 4 columns if present in file
    missing = [c for c in FEEDBACK_COLUMNS_ORDER if c not in values]
    if missing:
        raise ValueError(
            "Feedback timestamp columns missing or unparseable in Timestamp.csv: " + ", ".join(missing)
        )

    return {k: values[k] for k in FEEDBACK_COLUMNS_ORDER}


def _read_all_hhmm_candidates(*, session: SessionIdentity, data_dir: Path) -> Dict[str, int]:
    """Read all HHMM-looking values from the Timestamp.csv row.

    Used to estimate a constant clock offset between device-recorded timestamps
    and the experimenter's HHMM log for that day.
    """
    ts_path = _find_timestamp_csv(data_dir)
    ts_df = pd.read_csv(ts_path)
    row_df = ts_df[(ts_df["Date"] == session.date_str) & (ts_df["Subject"] == session.subject_id)]
    if row_df.empty:
        return {}
    row = row_df.iloc[0]

    out: Dict[str, int] = {}
    for col in ts_df.columns:
        if col in {"Date", "Subject"}:
            continue
        hhmm = _clean_hhmm_cell(row[col])
        if hhmm is None:
            continue
        # HHMM plausibility check
        hour = hhmm // 100
        minute = hhmm % 100
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            out[str(col)] = hhmm
    return out


def _estimate_clock_offset(*, extracted_csv_path: Path, date_str: str, earliest_hhmm: int) -> timedelta:
    """Estimate a constant offset to align HHMM log to device Timestamp.

    We align:
      datetime(date, earliest_hhmm)  ->  min(Timestamp in extracted CSV)

    This handles sessions where the acquisition device clock is shifted.
    """
    skiprows = _detect_csv_header_skiprows(extracted_csv_path)
    df_head = pd.read_csv(extracted_csv_path, skiprows=skiprows, nrows=200_000)
    if "Timestamp" not in df_head.columns:
        return timedelta(0)
    ts = pd.to_datetime(df_head["Timestamp"], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return timedelta(0)
    ts_min = ts.min().to_pydatetime()

    earliest_dt = _hhmm_to_datetime(date_str, earliest_hhmm)
    return ts_min - earliest_dt


def split_feedback_segments(
    *,
    session_id: str,
    extracted_csv_path: Path,
    buffer_sec: float = 15.0,
    measure_sec: float = 60.0,
    chunksize: int = 400_000,
    allow_missing: bool = False,
) -> Tuple[SessionIdentity, Dict[str, Path], list[str]]:
    """Create 4 padded segments for Feedback windows.

    Returns:
        (SessionIdentity, {segment_key: segment_csv_path}, missing_keys)
    """
    data_dir = PROJECT_ROOT / "Data"
    session = resolve_session_identity(session_id=session_id, data_dir=data_dir)

    if not extracted_csv_path.exists():
        raise FileNotFoundError(f"Extracted CSV not found: {extracted_csv_path}")

    feedback_hhmm = _read_feedback_hhmm_values(session=session, data_dir=data_dir)

    # Estimate constant clock offset if the device timestamp clock is shifted.
    hhmm_candidates = _read_all_hhmm_candidates(session=session, data_dir=data_dir)
    clock_offset = timedelta(0)
    if hhmm_candidates:
        earliest_hhmm = min(hhmm_candidates.values())
        clock_offset = _estimate_clock_offset(
            extracted_csv_path=extracted_csv_path,
            date_str=session.date_str,
            earliest_hhmm=earliest_hhmm,
        )

    segments_dir = data_dir / "Processed" / session_id / "split_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Prepare time windows
    segment_windows: Dict[str, Tuple[datetime, datetime, datetime, datetime]] = {}
    for key, hhmm in feedback_hhmm.items():
        start_dt = _hhmm_to_datetime(session.date_str, hhmm) + clock_offset
        original_start = start_dt
        original_end = start_dt + timedelta(seconds=measure_sec)
        padded_start = start_dt - timedelta(seconds=buffer_sec)
        padded_end = start_dt + timedelta(seconds=measure_sec + buffer_sec)
        segment_windows[key] = (padded_start, padded_end, original_start, original_end)

    # Write each segment by scanning the extracted CSV once per segment.
    # This is simple and robust; performance is acceptable for the expected dataset sizes.
    out_paths: Dict[str, Path] = {}
    missing_keys: list[str] = []

    for idx, key in enumerate(FEEDBACK_COLUMNS_ORDER, start=1):
        padded_start, padded_end, original_start, original_end = segment_windows[key]
        out_path = segments_dir / f"{idx:02d}_{key}.csv"
        if out_path.exists():
            out_path.unlink()

        wrote_header = False
        total_rows = 0

        skiprows = _detect_csv_header_skiprows(extracted_csv_path)
        for chunk in pd.read_csv(extracted_csv_path, chunksize=chunksize, skiprows=skiprows):
            if "Timestamp" not in chunk.columns:
                raise ValueError(
                    f"Extracted CSV has no 'Timestamp' column: {extracted_csv_path}. "
                    "Cannot align with HHMM feedback timestamps."
                )

            ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
            mask = (ts >= padded_start) & (ts <= padded_end)
            if not mask.any():
                continue

            seg = chunk.loc[mask].copy()
            seg["_original_start_ts"] = original_start.isoformat(sep=" ")
            seg["_original_end_ts"] = original_end.isoformat(sep=" ")

            seg.to_csv(out_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            total_rows += len(seg)

        if total_rows == 0:
            if allow_missing:
                missing_keys.append(key)
                # Ensure we don't leave empty files around.
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                continue
            raise ValueError(
                f"No rows extracted for segment {key} ({padded_start}..{padded_end}). "
                "Check Recording Time Stamp metadata and Timestamp alignment."
            )

        out_paths[key] = out_path

    if not out_paths:
        raise ValueError("No Feedback segments could be extracted for this session.")

    return session, out_paths, missing_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Split extracted ECG into 4 Feedback resting-HR segments.")
    parser.add_argument("--session", required=True, help="Session ID like 251216_TK")
    parser.add_argument("--extracted", type=Path, required=True, help="Path to extracted *_ext.csv")
    parser.add_argument("--buffer-sec", type=float, default=15.0)
    parser.add_argument("--measure-sec", type=float, default=60.0)
    args = parser.parse_args()

    session, out_paths = split_feedback_segments(
        session_id=args.session,
        extracted_csv_path=args.extracted,
        buffer_sec=args.buffer_sec,
        measure_sec=args.measure_sec,
    )

    print(f"Session: {session.session_id} ({session.subject_id}, BF_Type={session.bf_type})")
    for k, p in out_paths.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
