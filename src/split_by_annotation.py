#!/usr/bin/env python3
"""
アノテーションファイルに従ってCSVファイルを分割するプログラム

使用例:
    python3 split_by_annotation.py <分割対象ファイル> <アノテーションファイル>
    
    例:
    python3 split_by_annotation.py data.csv annotation.csv
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Tuple, Any
import re

# Import configuration for consistency
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_pipeline.config import Config

# Get default config instance
_config = Config()

# Edge artifact handling: add padding to prevent filter artifacts at segment boundaries
# This padding will be removed after filtering in the processing pipeline
FILTER_PADDING_SEC = _config.FILTER_PADDING_SEC  # seconds
SAMPLING_RATE = _config.SAMPLING_RATE  # Hz


# 分割対象のアノテーションペア（Start/End）
ANNOTATION_PAIRS = [
    ("GoNoGo_Baseline_Practice_Start", "GoNoGo_Baseline_Practice_End"),
    ("GoNoGo_Baseline_Start", "GoNoGo_Baseline_End"),
    ("Resting_HR_1_Set1_Start", "Resting_HR_1_Set1_End"),
    ("Session_Start", "Session_Stop"),
    ("Resting_HR_2_Set1_Start", "Resting_HR_2_Set1_End"),
    ("GoNoGo_Set1_Start", "GoNoGo_Set1_End"),
    ("Resting_HR_1_Set2_Start", "Resting_HR_1_Set2_End"),
    ("Resting_HR_2_Set2_Start", "Resting_HR_2_Set2_End"),
    ("GoNoGo_Set2_Start", "GoNoGo_Set2_End"),
]


def read_csv_with_metadata_skip(file_path: str) -> pd.DataFrame:
    """
    CSVファイルを読み込む際、先頭のメタデータ行を自動的にスキップする
    
    Args:
        file_path: 読み込むCSVファイルのパス
        
    Returns:
        読み込まれたDataFrame
    """
    # まず先頭数行を読んで、どこからデータが始まるかを判定
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(10)]
    
    skip_rows = 0
    for i, line in enumerate(lines):
        # カンマ区切りのヘッダー行を探す
        # "Time (s)" や "Timestamp" などの典型的なヘッダーを含む行を探す
        if 'Timestamp' in line or 'Time (s)' in line:
            skip_rows = i
            break
    
    # データを読み込む
    if skip_rows > 0:
        print(f"  メタデータ行を検出: {skip_rows}行をスキップします")
        df = pd.read_csv(file_path, skiprows=skip_rows)
    else:
        df = pd.read_csv(file_path)
    
    return df


def find_all_timestamps_for_annotations(
    df_annotation: pd.DataFrame, start_annotation: str, end_annotation: str
) -> list[Tuple[Any, Any]]:
    """
    指定されたアノテーションペアのタイムスタンプを全て取得する
    
    Args:
        df_annotation: アノテーションDataFrame
        start_annotation: 開始アノテーション名
        end_annotation: 終了アノテーション名
        
    Returns:
        [(start_timestamp, end_timestamp), ...] のリスト
        見つからない場合は空リスト
    """
    start_rows = df_annotation[df_annotation['Annotation'] == start_annotation]
    end_rows = df_annotation[df_annotation['Annotation'] == end_annotation]
    
    if start_rows.empty or end_rows.empty:
        return []
    
    # 各開始アノテーションに対応する終了アノテーションを見つける
    result = []
    for _, start_row in start_rows.iterrows():
        start_timestamp = start_row['Timestamp']
        
        # この開始タイムスタンプより後の終了アノテーションを探す
        matching_end = end_rows[end_rows['Timestamp'] > start_timestamp]
        
        if not matching_end.empty:
            # 最初に見つかった終了アノテーションを使用
            end_timestamp = matching_end.iloc[0]['Timestamp']
            result.append((start_timestamp, end_timestamp))
    
    return result


def _coerce_timestamp_column(df: pd.DataFrame, column: str = "Timestamp") -> Tuple[pd.DataFrame, str]:
    """Timestamp列を比較・ソートしやすい型に揃える。

    優先順位:
      1) datetime (ISO形式など)
      2) numeric
      3) fallback: そのまま (string等)

    Returns:
        (df, kind) kind は 'datetime' | 'numeric' | 'raw'
    """
    if column not in df.columns:
        return df, "raw"

    series = df[column]

    dt = pd.to_datetime(series, errors="coerce")
    dt_valid_ratio = float(dt.notna().mean()) if len(series) else 0.0
    if dt_valid_ratio >= 0.95:
        df = df.copy()
        df[column] = dt
        return df, "datetime"

    num = pd.to_numeric(series, errors="coerce")
    num_valid_ratio = float(num.notna().mean()) if len(series) else 0.0
    if num_valid_ratio >= 0.95:
        df = df.copy()
        df[column] = num
        return df, "numeric"

    return df, "raw"


def calculate_timestamp_offset(df: pd.DataFrame, timestamp_col: str, offset_sec: float) -> Any:
    """
    Calculate a timestamp offset based on the average sampling interval.
    
    Args:
        df: DataFrame containing timestamp data
        timestamp_col: Name of the timestamp column
        offset_sec: Offset in seconds
        
    Returns:
        Timestamp offset value (numeric or datetime)
    """
    ts_series = df[timestamp_col]
    
    # Check if timestamps are numeric
    if pd.api.types.is_numeric_dtype(ts_series):
        # For numeric timestamps, just add the offset
        return offset_sec
    
    # For datetime timestamps, use timedelta
    try:
        # Calculate average sampling interval from first 100 samples
        sample_size = min(100, len(ts_series))
        if sample_size >= 2:
            intervals = ts_series.iloc[:sample_size].diff().dropna()
            avg_interval = intervals.mean()
            
            # Calculate number of samples corresponding to the offset duration
            avg_interval_sec = avg_interval.total_seconds()
            if avg_interval_sec > 0:
                n_samples = int(offset_sec / avg_interval_sec)
                return avg_interval * n_samples
            else:
                # Fallback: use timedelta if the average interval is invalid
                return pd.Timedelta(seconds=offset_sec)
        else:
            # Fallback: use timedelta
            return pd.Timedelta(seconds=offset_sec)
    except Exception:
        # If datetime operations fail, use timedelta
        return pd.Timedelta(seconds=offset_sec)


def split_csv_by_annotations(target_file: str, annotation_file: str) -> None:
    """
    アノテーションファイルに従ってCSVファイルを分割する
    
    Args:
        target_file: 分割対象のCSVファイル
        annotation_file: アノテーション情報を含むCSVファイル
    """
    # ファイルを読み込む
    print(f"分割対象ファイル: {target_file}")
    print(f"アノテーションファイル: {annotation_file}")
    print()
    
    # メタデータ行の検出とスキップ
    # ファイルの先頭数行をチェックして、メタデータ行をスキップする
    df_target = read_csv_with_metadata_skip(target_file)
    df_annotation = read_csv_with_metadata_skip(annotation_file)
    
    print(f"分割対象ファイルの行数: {len(df_target)}")
    print(f"アノテーションファイルの行数: {len(df_annotation)}")
    print()
    
    # カラム確認
    if 'Timestamp' not in df_target.columns:
        raise ValueError(f"分割対象ファイルに Timestamp 列が見つかりません: {target_file}")
    if 'Timestamp' not in df_annotation.columns or 'Annotation' not in df_annotation.columns:
        raise ValueError(f"アノテーションファイルに必要な列が見つかりません: {annotation_file}")

    # Timestamp列を比較・ソートに向いた型へ揃える
    df_target, target_ts_kind = _coerce_timestamp_column(df_target, "Timestamp")
    df_annotation, ann_ts_kind = _coerce_timestamp_column(df_annotation, "Timestamp")
    if target_ts_kind != ann_ts_kind and target_ts_kind != "raw" and ann_ts_kind != "raw":
        print(
            f"  ⚠ Timestamp型が一致しません (target={target_ts_kind}, annotation={ann_ts_kind})。\n"
            f"    可能な範囲で処理しますが、抽出が空になる場合はTimestampの形式を確認してください。"
        )
    
    # 出力ディレクトリの準備
    target_path = Path(target_file)
    output_dir = target_path.parent / "split_segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 以前の実行結果が残っていると紛らわしいので、既存の分割CSVを削除
    # 対象: "01_xxx.csv" のようなファイル名のみ
    existing_segment_files = [
        p for p in output_dir.glob("*.csv") if re.match(r"^\d{2}_.+\.csv$", p.name)
    ]
    if existing_segment_files:
        for p in existing_segment_files:
            try:
                p.unlink()
            except Exception:
                # 消せないファイルがあっても処理は継続
                pass
        print(f"既存の分割ファイルを削除: {len(existing_segment_files)} 件")
    
    print(f"出力ディレクトリ: {output_dir}")
    print()
    
    # --- 1) 全セグメントを収集 (複数出現も含む) ---
    segments: list[dict[str, Any]] = []
    missing_pairs = 0
    for pair_idx, (start_annotation, end_annotation) in enumerate(ANNOTATION_PAIRS, 1):
        print(f"[{pair_idx}/{len(ANNOTATION_PAIRS)}] {start_annotation} → {end_annotation}")

        timestamp_pairs = find_all_timestamps_for_annotations(df_annotation, start_annotation, end_annotation)
        if not timestamp_pairs:
            print("  ⚠ アノテーションが見つかりませんでした。スキップします。")
            print()
            missing_pairs += 1
            continue

        # 出現順がTimestamp順になるように揃える（安全策）
        timestamp_pairs_sorted = sorted(timestamp_pairs, key=lambda x: x[0])
        occurrence_count = len(timestamp_pairs_sorted)
        print(f"  検出された出現回数: {occurrence_count}")

        segment_name = start_annotation.replace("_Start", "")
        for occurrence_idx, (start_ts, end_ts) in enumerate(timestamp_pairs_sorted, 1):
            if occurrence_count > 1:
                print(f"  [{occurrence_idx}/{occurrence_count}]")

            print(f"    開始タイムスタンプ: {start_ts}")
            print(f"    終了タイムスタンプ: {end_ts}")

            segments.append(
                {
                    "segment_name": segment_name,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "occurrence_idx": occurrence_idx,
                    "occurrence_count": occurrence_count,
                }
            )

        print()

    if not segments:
        print("完了: 分割対象のセグメントが見つかりませんでした。")
        return

    # --- 2) 開始時刻でソートし、通し番号(インデックス)を振る ---
    segments_sorted = sorted(segments, key=lambda s: s["start_ts"])

    # --- 3) 通し番号順に抽出して保存 ---
    split_count = 0
    total = len(segments_sorted)
    print(f"保存処理: {total} セグメント（開始時刻順）")
    print(f"フィルタパディング: {FILTER_PADDING_SEC}秒")
    print()
    
    # Calculate timestamp offset for padding
    padding_offset = calculate_timestamp_offset(df_target, "Timestamp", FILTER_PADDING_SEC)

    for global_idx, seg in enumerate(segments_sorted, 1):
        segment_name = seg["segment_name"]
        start_ts = seg["start_ts"]
        end_ts = seg["end_ts"]
        occurrence_idx = int(seg["occurrence_idx"])
        occurrence_count = int(seg["occurrence_count"])

        print(f"[{global_idx}/{total}] {segment_name}")
        print(f"    開始タイムスタンプ: {start_ts}")
        print(f"    終了タイムスタンプ: {end_ts}")
        
        # Add padding to start and end timestamps
        start_ts_padded = start_ts - padding_offset
        end_ts_padded = end_ts + padding_offset
        
        print(f"    パディング後開始: {start_ts_padded}")
        print(f"    パディング後終了: {end_ts_padded}")

        # Extract segment with padding
        df_segment = df_target[
            (df_target["Timestamp"] >= start_ts_padded)
            & (df_target["Timestamp"] <= end_ts_padded)
        ]

        if df_segment.empty:
            print("    ⚠ 該当するデータが見つかりませんでした。スキップします。")
            print()
            continue

        print(f"    抽出行数（パディング含む): {len(df_segment)}")
        
        # Add metadata columns to track original (unpadded) segment boundaries
        # These will be used in the processing pipeline to trim the padding after filtering
        df_segment = df_segment.copy()
        df_segment['_original_start_ts'] = start_ts
        df_segment['_original_end_ts'] = end_ts

        # ファイル名: 通し番号_セグメント名(_出現番号).csv
        # 例: 04_Session_01.csv, 07_Session_02.csv
        if occurrence_count > 1:
            output_filename = f"{global_idx:02d}_{segment_name}_{occurrence_idx:02d}.csv"
        else:
            output_filename = f"{global_idx:02d}_{segment_name}.csv"

        output_path = output_dir / output_filename
        df_segment.to_csv(output_path, index=False)
        print(f"    ✓ 保存: {output_path}")
        print()

        split_count += 1

    print(f"完了: {split_count} 個のセグメントを分割しました。")


def main():
    """メイン関数"""
    if len(sys.argv) != 3:
        print("使用方法: python3 split_by_annotation.py <分割対象ファイル> <アノテーションファイル>")
        print()
        print("例:")
        print("  python3 split_by_annotation.py data.csv annotation.csv")
        sys.exit(1)
    
    target_file = sys.argv[1]
    annotation_file = sys.argv[2]
    
    # ファイルの存在確認
    if not Path(target_file).exists():
        print(f"エラー: 分割対象ファイルが見つかりません: {target_file}")
        sys.exit(1)
    
    if not Path(annotation_file).exists():
        print(f"エラー: アノテーションファイルが見つかりません: {annotation_file}")
        sys.exit(1)
    
    try:
        split_csv_by_annotations(target_file, annotation_file)
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
