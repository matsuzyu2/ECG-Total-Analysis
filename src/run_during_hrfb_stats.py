#!/usr/bin/env python3
"""Run paired statistical tests for During-HRFB delta (Last5 - First5).

This script consumes per-session outputs produced by:
  - src/run_during_hrfb_analysis.py

Data source:
  Results/<session_id>/during_hrfb_metrics.csv

Delta definition (per session, per condition):
  delta = time_mean_hr(Last5) - time_mean_hr(First5)

Tests (paired within subject/session):
  - BF_Type = Inc: Target delta vs Control delta (paired)
  - BF_Type = Dec: Target delta vs Control delta (paired)

Method selection:
  - Shapiro-Wilk normality check on paired differences (Target - Control)
  - If normality p >= alpha: paired t-test
  - Else: Wilcoxon signed-rank test

Output:
  - Results/during_hrfb_stats_report.md

Console output is intentionally minimal.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Ensure scripts work when executed from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_group_stats import _paired_test


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_session_metrics_csvs(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        return []
    for session_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        csv_path = session_dir / "during_hrfb_metrics.csv"
        if csv_path.exists():
            yield csv_path


def _read_all_metrics(results_dir: Path) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for p in _iter_session_metrics_csvs(results_dir):
        df = pd.read_csv(p)
        if "session_id" not in df.columns or df["session_id"].isna().all():
            df["session_id"] = p.parent.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _compute_session_condition_deltas(all_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session deltas for Control/Target.

    Returns wide table with columns:
      session_id, Subject, BF_Type, Control, Target

    Where Control/Target are deltas (Last5 - First5).
    """
    if all_df.empty:
        return pd.DataFrame(columns=["session_id", "Subject", "BF_Type", "Control", "Target"])

    required = {"session_id", "Subject", "BF_Type", "condition", "phase", "time_mean_hr"}
    missing = required - set(all_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in during_hrfb_metrics.csv data: {sorted(missing)}")

    df = all_df.copy()
    df["time_mean_hr"] = _to_numeric_series(df["time_mean_hr"])
    df["phase"] = df["phase"].astype(str)
    df["condition"] = df["condition"].astype(str)

    df = df[df["phase"].isin(["First5", "Last5"])].copy()
    df = df[df["condition"].isin(["Control", "Target"])].copy()

    # Deduplicate within each session/condition/phase by taking the first non-null value.
    df = df.sort_values(["session_id", "condition", "phase"]).copy()

    gcols = ["session_id", "Subject", "BF_Type", "condition", "phase"]
    df = (
        df.groupby(gcols, as_index=False)["time_mean_hr"]
        .agg(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan)
        .copy()
    )

    # Pivot phase -> columns, compute delta
    wide_phase = df.pivot_table(
        index=["session_id", "Subject", "BF_Type", "condition"],
        columns="phase",
        values="time_mean_hr",
        aggfunc="first",
    ).reset_index()

    wide_phase["delta"] = wide_phase.get("Last5") - wide_phase.get("First5")

    # Pivot condition -> columns
    out = wide_phase.pivot_table(
        index=["session_id", "Subject", "BF_Type"],
        columns="condition",
        values="delta",
        aggfunc="first",
    ).reset_index()

    # Ensure both columns exist
    for col in ["Control", "Target"]:
        if col not in out.columns:
            out[col] = np.nan

    return out[["session_id", "Subject", "BF_Type", "Control", "Target"]]


def _fmt_mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "N/A"
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if x.shape[0] > 1 else float("nan")
    if np.isfinite(sd):
        return f"{mean:.2f} ± {sd:.2f}"
    return f"{mean:.2f}"


def _fmt_float(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{digits}f}"


def _group_label(bf_type: str) -> str:
    if bf_type == "Inc":
        return "Increase (Inc)"
    if bf_type == "Dec":
        return "Decrease (Dec)"
    return bf_type or "Unknown"


def _run_group_test(wide: pd.DataFrame, *, bf_type: str, alpha: float) -> dict:
    d = wide[wide["BF_Type"] == bf_type].copy()

    # Paired rows only: both deltas must exist
    ok = pd.notna(d["Control"]) & pd.notna(d["Target"]) & np.isfinite(d["Control"]) & np.isfinite(d["Target"])
    d = d[ok].copy()

    x = d["Control"].to_numpy(dtype=float)
    y = d["Target"].to_numpy(dtype=float)

    res = _paired_test(x=x, y=y, alpha=alpha)

    p_val = res.get("p")
    significant = False
    try:
        significant = bool(np.isfinite(float(p_val)) and float(p_val) < alpha)
    except Exception:
        significant = False

    return {
        "bf_type": bf_type,
        "group": _group_label(bf_type),
        "n_pairs": int(res.get("n", 0)),
        "control_mean_sd": _fmt_mean_sd(pd.Series(x)),
        "target_mean_sd": _fmt_mean_sd(pd.Series(y)),
        "mean_diff": _fmt_float(res.get("mean_diff"), digits=3),
        "normality_p": _fmt_float(res.get("normality_p"), digits=4),
        "test": str(res.get("test", "N/A")),
        "stat": _fmt_float(res.get("stat"), digits=4),
        "p": _fmt_float(res.get("p"), digits=6),
        "judgement": "有意差あり" if significant else "有意差なし",
    }


def _render_report(*, session_count: int, wide: pd.DataFrame, inc_row: dict, dec_row: dict, alpha: float) -> str:
    today = date.today().isoformat()

    lines: list[str] = []
    lines.append("# 心拍FB中（During HRFB）統計レポート")
    lines.append("")
    lines.append(f"Date: {today}")
    lines.append("")

    lines.append("## 概要")
    lines.append("本レポートは、心拍FB中の心拍変化量（ΔHR）が Target 条件と Control 条件で異なるかを検定します。")
    lines.append("")
    lines.append("- データソース: `Results/<session_id>/during_hrfb_metrics.csv`")
    lines.append("- ΔHR 定義（セッション・条件ごと）: `ΔHR = time_mean_hr(Last5) − time_mean_hr(First5)`（単位: bpm）")
    lines.append("- 比較は被験者内対応（Paired）: Target の ΔHR vs Control の ΔHR")
    lines.append("")

    lines.append("## 検定手法")
    lines.append("- 正規性確認: 対応差（Target − Control）に対する Shapiro-Wilk 検定")
    lines.append(f"- 判定ルール（alpha={alpha:.2f}）:")
    lines.append("  - 正規性 p ≥ alpha → 対応のある t 検定（paired t-test）")
    lines.append("  - それ以外 → Wilcoxon 符号付順位検定")
    lines.append("")
    lines.append(f"処理対象セッション数（during_hrfb_metrics.csv が存在）: {session_count}")
    lines.append("")

    lines.append("## 結果")
    lines.append("")
    lines.append("| 群（BF_Type） | N（ペア数） | Control ΔHR（平均±SD） | Target ΔHR（平均±SD） | 平均差（Target−Control） | Shapiro p | 検定 | 統計量 | p値 | 判定（p<0.05） |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---|")

    for r in [inc_row, dec_row]:
        lines.append(
            "| {group} | {n_pairs} | {control_mean_sd} | {target_mean_sd} | {mean_diff} | {normality_p} | {test} | {stat} | {p} | {judgement} |".format(**r)
        )

    lines.append("")
    lines.append("### 補足")
    lines.append("- paired解析のため、Target と Control の ΔHR が両方そろっているセッションのみを用います。")
    lines.append("- `検定` は Shapiro-Wilk の結果に基づき自動選択されます（paired t-test / Wilcoxon）。")

    return "\n".join(lines) + "\n"


def run(*, results_dir: Path, output_md: Path, alpha: float) -> None:
    all_df = _read_all_metrics(results_dir)

    session_count = len(list(_iter_session_metrics_csvs(results_dir)))

    wide = _compute_session_condition_deltas(all_df)

    inc_row = _run_group_test(wide, bf_type="Inc", alpha=alpha)
    dec_row = _run_group_test(wide, bf_type="Dec", alpha=alpha)

    report = _render_report(session_count=session_count, wide=wide, inc_row=inc_row, dec_row=dec_row, alpha=alpha)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="During HRFB paired stats: Target vs Control (Inc/Dec).")
    parser.add_argument("--results-dir", type=Path, default=_project_root() / "Results")
    parser.add_argument("--output", type=Path, default=_project_root() / "Results" / "during_hrfb_stats_report.md")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    # Minimal console output (per requirement)
    session_count = len(list(_iter_session_metrics_csvs(args.results_dir)))
    print(f"Sessions processed: {session_count}")

    run(results_dir=args.results_dir, output_md=args.output, alpha=float(args.alpha))

    print("Statistical tests completed.")

    try:
        rel = args.output.resolve().relative_to(_project_root())
        out_display = str(rel)
    except Exception:
        out_display = str(args.output)
    print(f"Saved report to: {out_display}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
