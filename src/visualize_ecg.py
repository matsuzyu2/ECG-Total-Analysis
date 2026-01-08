"""Visualize ECG signals stored in the extracted CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.graph_objects as go

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "Data"/"Processed"/"251210_MS"/"split_segments"
DEFAULT_ECG_FILE: Final[Path] = DATA_DIR / "01_Session_01.csv"
EXPECTED_COLUMNS: Final[set[str]] = {"Time (s)", "ExGa 1(uV)"}


def load_ecg_data(file_path: Path) -> pd.DataFrame:
    """Load the ECG CSV, ensure the expected schema, and normalize time."""

    resolved_path = file_path.expanduser().resolve()
    return (
        pd.read_csv(resolved_path)
        .pipe(_ensure_columns, file_path=resolved_path)
        .pipe(_rename_columns)
        .pipe(_normalize_time)
        .pipe(_sort_by_time)
    )


def _ensure_columns(df: pd.DataFrame, *, file_path: Path) -> pd.DataFrame:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "対象のCSVは 251215_ECG_TK_01_extracted.csv と同じフォーマットである必要があります. "
            f"不足している列: {', '.join(missing)} (source: {file_path})."
        )
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "Time (s)": "time_seconds",
            "ExGa 1(uV)": "ecg_value_uv",
        }
    )


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    baseline = df["time_seconds"].iloc[0]
    return df.assign(time_seconds=df["time_seconds"] - baseline)


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("time_seconds").reset_index(drop=True)


def create_figure(df: pd.DataFrame) -> go.Figure:
    """Generate an interactive Plotly line chart for the ECG waveform."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time_seconds"],
            y=df["ecg_value_uv"],
            mode="lines",
            name="ExGa 1",
            line=dict(color="#2E86AB", width=1.5),
        )
    )

    # The layout balances detailed inspection and high-level trends.
    fig.update_layout(
        title="ECG Data Visualization",
        xaxis=dict(
            title=dict(text="時刻 (秒)", font=dict(size=16, color="#34495E")),
            showgrid=True,
            gridcolor="#ECF0F1",
            zeroline=True,
            zerolinecolor="#BDC3C7",
            zerolinewidth=2,
        ),
        yaxis=dict(
            title=dict(text="ExGa 1 (μV)", font=dict(size=16, color="#34495E")),
            showgrid=True,
            gridcolor="#ECF0F1",
            zeroline=True,
            zerolinecolor="#BDC3C7",
            zerolinewidth=2,
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        width=1600,
        height=600,
        margin=dict(l=80, r=40, t=80, b=60),
    )
    return fig


def summarize_data(df: pd.DataFrame) -> None:
    """Print simple stats so users can sanity-check the dataset."""

    duration = df["time_seconds"].iloc[-1] if not df.empty else 0.0
    print("CSVファイルの読み込みに成功しました。")
    print(f"データポイント数: {len(df)}")
    print(f"計測時間: {duration:.2f} 秒")


def main() -> None:
    """CLI entry point that loads the ECG and renders the figure."""

    parser = argparse.ArgumentParser(
        description="ECGデータを可視化します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"例:\n"
        f"  python {Path(__file__).name}\n"
        f"  python {Path(__file__).name} --file Data/custom_ecg.csv",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=DEFAULT_ECG_FILE,
        help=f"解析対象のECG CSVファイルパス (デフォルト: {DEFAULT_ECG_FILE.relative_to(DATA_DIR.parent)})",
    )
    args = parser.parse_args()

    try:
        df = load_ecg_data(args.file)
    except FileNotFoundError:
        print(f"エラー: ファイル '{args.file}' が見つかりません。")
        print("正しいファイルパスを指定してください。")
        return
    except ValueError as error:
        print(f"エラー: {error}")
        return

    summarize_data(df)
    create_figure(df).show()


if __name__ == "__main__":
    main()