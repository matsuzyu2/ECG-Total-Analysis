# 安静時心拍フィードバック解析（中央値ベース）仕様書

- 対象リポジトリ: ECG_Total_Analysis
- 対象機能: 「安静時心拍フィードバック解析（Resting HR Feedback）」および「Responder群間比較」
- 最終更新: 2026-01-14
- 参照実装:
  - `src/run_hrv_metrics.py`
  - `src/run_resting_hr_feedback_analysis.py`
  - `src/report_good_responders.py`
  - `src/plot_good_vs_non_good.py`
  - （中央値HR算出の根拠として）`src/analysis_pipeline/hrv.py`
  - （セグメント切り出しの根拠として）`src/split_by_feedback_timestamp.py`

---

## 1. 概要 (Overview)

### 1.1 解析の目的
本解析は、安静時心拍（Resting HR）に対するバイオフィードバック（Biofeedback; BF）の介入効果を、**Control条件とTarget条件のPre/Post**から定量化し、
- セッション単位の介入効果の可視化
- Difference-in-Differences（DID）に基づくGood Responder判定
- Good vs Non-Goodの群間比較（統計検定含む）
を実行することを目的とする。

### 1.2 主要な変更点（Mean → Median）
従来の代表値 `time_mean_hr`（平均HR）は外れ値（ノイズ、誤検出ピーク、局所的アーチファクト等）の影響を受けやすい。
このため、本実装では**中央値HR `time_median_hr`** を主要な代表値として採用し、
- `run_resting_hr_feedback_analysis.py` の可視化
- `report_good_responders.py` のGood Responder判定
- `plot_good_vs_non_good.py` の群間比較
において **Median優先（無い場合のみMeanにフォールバック）** のロジックに統一した。

> 後方互換: 既存CSVに `time_median_hr` が存在しない場合、`time_mean_hr` を使用する。

---

## 2. 解析パイプライン (Analysis Pipeline)

### 2.1 入力データ (Input Data)

#### 2.1.1 生データ
- ルート: `Data/Raw/{session_id}/*.txt`
- `run_resting_hr_feedback_analysis.py` はセッションフォルダ内の **複数txt** を許容し、抽出CSVへ順次追記する。

#### 2.1.2 被験者メタデータ
- `Data/Subject/SubjectData.csv`
- 主に使用する列:
  - `Date`（`YYYY_MM_DD`）
  - `Subject`（例: `TK`, `MS_1` など）
  - `BF_Type`（`Inc` or `Dec`）

セッションIDは `YYMMDD_{token}`（例: `251216_TK`）形式。
`split_by_feedback_timestamp.resolve_session_identity()` が `SubjectData.csv` を参照して、
- `Date` を `YYMMDD -> YYYY_MM_DD` に変換
- `token` と `Subject` の照合（完全一致優先、次にprefix一致）
を用いて、`Subject` と `BF_Type` を解決する。

#### 2.1.3 タイムスタンプ（Feedback開始HHMM）
- `Data/Timestamp/Timestamp_2.csv`
- 行キー: `Date` と `Subject`
- 主要列（固定順序）:
  - `Feedback_Con_pre`
  - `Feedback_Con_post`
  - `Feedback_pre`
  - `Feedback_post`

各セルはHHMM（例: `1438`）として解釈される。
CSVに `"""1438"""` のような引用符が混入するケースを想定し、引用符除去後にdigitsのみを許容する。

---

### 2.2 ステップ別処理

本パイプラインは `src/run_resting_hr_feedback_analysis.py` がエンドツーエンドで実行する。

#### Step 0: 抽出CSVの作成（Raw → extracted）
関数: `ensure_extracted_csv(session_id, rebuild)`

- 入力: `Data/Raw/{session_id}/*.txt`
- 出力: `Data/Processed/{session_id}/extracted/{session_id}_ext.csv`
- 実装:
  - `extract_ecg_columns.extract_columns()` を呼び出し
  - 複数txtがある場合は同一出力CSVに追記
  - `--rebuild-extracted` 指定時は既存CSVを削除して再生成

#### Step 1: Feedbackセグメント切り出し（extracted → split_segments）
関数: `split_feedback_segments()`（`src/split_by_feedback_timestamp.py`）

- 目的: Control/Target × Pre/Post の4窓（各60秒）を切り出す。
- 重要仕様:
  - **フィルタ端点アーチファクト対策**として、各窓に **±buffer秒のパディング**を付与して書き出す。
  - 書き出しCSVに、元の測定区間（パディング除去後の区間）を示すメタ列を追加:
    - `_original_start_ts`
    - `_original_end_ts`

**時間窓定義**（HHMM基準）:
- 測定区間:
  - `original_start = HHMM_start`
  - `original_end   = original_start + measure_sec`
- 書き出し区間（パディング付き）:
  - `padded_start = original_start - buffer_sec`
  - `padded_end   = original_start + measure_sec + buffer_sec`

デフォルト値（呼び出し側）:
- `measure_sec = 60.0`
- `buffer_sec  = config.FILTER_PADDING_SEC`（通常15秒）

**デバイス時計ずれ補正（clock offset）**:
データの `Timestamp` 列（デバイス記録時刻）と、実験ログHHMMがずれているセッションに対し、一定オフセットを推定して補正する。
- `Timestamp_2.csv` の当該行に含まれるHHMM候補の最小値 `earliest_hhmm` を取得
- `extracted CSV` の `Timestamp` 列の最小値 `ts_min` と
  `datetime(date, earliest_hhmm)` を整列させ、
  `clock_offset = ts_min - earliest_dt`
を全Feedback窓の開始時刻に加算する。

**切り出し条件**:
- `extracted CSV` に `Timestamp` 列が無い場合はエラー
- 該当窓に一致する行が0の場合:
  - `allow_missing=False`: 例外（セッション処理失敗）
  - `allow_missing=True`: missingとしてスキップし、空ファイルは残さない

出力ファイル名:
- `Data/Processed/{session_id}/split_segments/01_Feedback_Con_pre.csv`
- `Data/Processed/{session_id}/split_segments/02_Feedback_Con_post.csv`
- `Data/Processed/{session_id}/split_segments/03_Feedback_pre.csv`
- `Data/Processed/{session_id}/split_segments/04_Feedback_post.csv`

#### Step 2: 診断・前処理・Rピーク検出（split_segments → peaks_*.json）
関数: `process_segment()`（`src/run_signal_diagnosis.py`）

本仕様書の主対象ではないが、Resting HR解析の前提となるため概要のみ記載する。

- ロード: `analysis_pipeline.io_utils.load_segment_csv()`
  - µV→mVの自動変換を含む
  - パディング有無を判定し、`_original_start_ts/_original_end_ts` から
    `original_start_idx/original_end_idx` を推定する（下流のトリミングで使用）
- 極性診断: skewness計算
  - **自動反転は無効**（診断のみ）
- フィルタ: Butterworth bandpass（ゼロ位相）
- パディング除去: フィルタ後に `original_start_idx..original_end_idx` を保持してトリミング
- Rピーク検出: `analysis_pipeline.rpeak.detect_rpeaks()`

出力:
- `Results/{session_id}/diagnosis_{segment}.html`
- `Results/{session_id}/peaks_{segment}.json`

#### Step 3: HRV指標計算（peaks_*.json → HRV metrics）
関数: `process_peaks_file()`（`src/run_hrv_metrics.py`）

- 入力: `Results/{session_id}/peaks_{segment}.json`
- コア計算: `analysis_pipeline.hrv.compute_hrv_metrics()`
- RR生成: `rr_ms = diff(peak_times) * 1000`
- RRクリーニング: `clean_rr_intervals()`
  - 生理学的範囲フィルタ（`config.RR_MIN_MS..RR_MAX_MS`）
  - MAD（Median Absolute Deviation）による外れ値除去（`config.MAD_THRESHOLD`）

##### 2.2.3.1 `time_median_hr` の算出仕様
`analysis_pipeline/hrv.py` の `compute_hrv_metrics()` により以下で算出される。

1) Rピーク時刻 `peak_times`（秒）からRR（ms）を生成しクリーニング:
- $RR_i = (t_{i+1} - t_i) \times 1000$
- `rr_cleaned = clean_rr_intervals(RR).rr_cleaned`

2) HR系列（bpm）を生成:
- $HR_i = \frac{60000}{RR_i}$

3) 中央値:
- $time\_median\_hr = \mathrm{median}(HR)$

##### 2.2.3.2 中央30秒ウィンドウ（端点ノイズ低減）
HR関連フィールド（mean/median/std/min/max）は、可能なら **セグメント中央30秒**で再計算される。

適用条件:
- `segment_start_timestamp` が解釈可能
- `peak_timestamps` が `peak_times` と同数で存在

中央窓:
- セグメント長を `segment_dur` とし、中心を `segment_dur/2` とする
- 半窓幅を `min(15s, segment_dur/2)` として
  - `window_start = start + (segment_dur/2 - half_window)`
  - `window_end   = start + (segment_dur/2 + half_window)`

その範囲に入るピークのみでRR→HRを作り、再度 `clean_rr_intervals()` を適用した後に
`mean_hr/median_hr/std_hr/min_hr/max_hr` を上書きする。

> 60秒セグメントでは、結果として「両端15秒を除いた中央30秒」が用いられる。

中央窓が作れない場合:
- `compute_time_domain(rr_cleaned)` により全区間から算出された値がそのまま使用される。
- 品質メモに `"⚠ Mean HR computed from full 60s (central window unavailable)"` が追加される。

`run_hrv_metrics.py` のCSV出力には `time_mean_hr` と `time_median_hr` の両方が含まれる。

#### Step 4: Resting HR Feedback用のmetrics CSV生成
処理箇所: `run_session()`（`src/run_resting_hr_feedback_analysis.py`）

- 各セグメントについて `metrics.to_flat_dict()` を作成し、追加メタ列を付与して行にする。
- 追加される列（重要なもの）:
  - `segment_key`: `Feedback_Con_pre` など
  - `condition`: `control`（Con） / `target`（non-Con）
  - `phase`: `pre` / `post`
  - `BF_Type`: `SubjectData.csv` より
  - `Subject`: `SubjectData.csv` より
  - `Date`: `YYYY_MM_DD`

出力:
- `Results/{session_id}/resting_hr_feedback_metrics.csv`

#### Step 4.1: 追加の周波数解析（AR/Burg PSD; 任意）
処理箇所: `run_session()` 内（`src/run_resting_hr_feedback_analysis.py`）

- 目的: 60秒程度の短区間でWelch/FFTが不安定になりうるため、AR(Burg)で滑らかなPSDを得る。
- 使用RR:
  - **中心30秒ではなく、フルセグメントのRR**から生成（`peak_times`全体）
  - `rr_ms = diff(peak_times) * 1000` → `clean_rr_intervals()`
- パラメータ:
  - `fs_resample_hz=4.0`
  - `ar_order=16`
  - `nfft=1024`

出力:
- セッション内の断片を集約して1ページにまとめる:
  - `Results/{session_id}/hrv_psd_summary.html`

#### Step 5: Pre/Post可視化（Control vs Target）
関数: `_plot_pre_post(metrics_df, session_id, bf_type, output_html)`

- 可視化値:
  - 優先: `time_median_hr`
  - フォールバック: `time_mean_hr`

- 対象点:
  - Control: `control/pre`, `control/post`
  - Target: `target/pre`, `target/post`

- 表示:
  - 2系列（Control, Target）をmarkers+linesで表示
  - Δ注釈: `Δ = post - pre` を各系列に付与

出力:
- `Results/{session_id}/resting_hr_feedback_comparison.html`

#### Step 6: 全セッション一括処理（--all-sessions）
`run_resting_hr_feedback_analysis.py --all-sessions` の場合:
- `Data/Raw` 配下の全セッションを走査
- `allow_missing_segments=True` として、欠損セグメントがあっても継続
- 欠損レポートを逐次・最終書き出し:
  - `Results/resting_hr_feedback_missing_segments.csv`
- Increase/Decreaseごとに、各セッションのHTMLプロットをiframeで並べたサマリページを生成:
  - `Results/resting_hr_feedback_comparison_Increase.html`
  - `Results/resting_hr_feedback_comparison_Decrease.html`

---

## 3. レスポンダー判定ロジック (Responder Classification)

### 3.1 入力
判定は `src/report_good_responders.py` により実施する。

入力CSVの探索順序:
1. 優先: `Results/{session_id}/resting_hr_feedback_metrics.csv`
2. フォールバック:
   - `Data/Processed/{session_id}/metrics.csv`
   - `Data/Processed/{session_id}/resting_hr_feedback_metrics.csv`

各metrics CSVに必要な列:
- `condition`（`control`/`target`）
- `phase`（`pre`/`post`）
- `BF_Type`
- `Subject`
- かつHR列として `time_median_hr` または `time_mean_hr`

### 3.2 DID（Difference-in-Differences）定義
各セッション（=被験者）について、Median HRを用いて以下を計算する。

- $\Delta_{Control} = HR^{post}_{Control} - HR^{pre}_{Control}$
- $\Delta_{Target}  = HR^{post}_{Target}  - HR^{pre}_{Target}$
- $$Diff = \Delta_{Target} - \Delta_{Control}$$

ここで $HR$ は以下の優先順位で取得される。
- 優先: `time_median_hr`
- フォールバック: `time_mean_hr`

### 3.3 Good Respondersの定義
閾値を `Threshold (>=0)` とする（実装では `abs()` で正に正規化）。

- Decrease条件（`BF_Type = Dec`）:
  - $$Diff \le -Threshold$$
- Increase条件（`BF_Type = Inc`）:
  - $$Diff \ge Threshold$$

注意:
- `BF_Type` が `Inc/Dec` 以外（Unknown等）の場合は判定対象外。
- 必要なPre/Postが欠損し、`Diff` がNaNになる場合はGoodにしない。

### 3.4 出力
CLI:
- `python src/report_good_responders.py --threshold 0.0`
- `python src/report_good_responders.py --threshold 1.0 --output Results/good_responders.csv`

出力の挙動:
- `--output` 指定あり:
  - CSVを作成（主要列のみ）
  - 例: `Results/good_responders.csv`
- `--output` 指定なし:
  - stdoutに読みやすいテーブルを表示（Age/Sexがあれば追加表示）

CSV出力列（ユーザー向け）:
- `Subject`
- `session_id`
- `BF_Type`
- `control_delta`
- `target_delta`
- `diff`

ソート仕様（Goodのみ）:
- `Dec`: より負（小）ほど良い → `diff` 昇順
- `Inc`: より正（大）ほど良い → `diff` 降順

---

## 4. 群間比較と統計解析 (Group Comparison & Statistics)

本章は `src/plot_good_vs_non_good.py` の仕様である。

### 4.1 比較対象（比較に用いるデータ）
入力:
- `Results/{session_id}/resting_hr_feedback_metrics.csv`（全セッション分を走査）

必要列:
- `condition`, `phase`, `BF_Type`, `Subject`
- `time_median_hr` または `time_mean_hr`

各セッションにつき、Responder判定と同様に
- `control_delta`
- `target_delta`
- `diff`
を計算し、Good/Non-Goodに二分する。

### 4.2 解析に用いる比較指標（diff_plot）
本プロットでは、介入方向が異なる `Inc` と `Dec` を同一方向（「大きいほど良い」）に揃えるため、
可視化専用の指標 `diff_plot`（別名 `plot_diff`）を定義する。

- `BF_Type = Inc`:
  - `diff_plot = diff`
- `BF_Type = Dec`:
  - `diff_plot = -diff`（符号反転）

これにより、
- IncでもDecでも、**値が大きいほど介入効果が良い**
という解釈が可能になる。

> 注意: Responder判定そのもの（Good/Non-Good）は `diff` に対して行う。`diff_plot` は可視化と群間検定のための整列指標。

### 4.3 統計検定
検定:
- **Mann–Whitney U検定（独立2群、両側検定）**
- 実装: `scipy.stats.mannwhitneyu(x, y, alternative="two-sided")`

入力データ:
- `x = diff_plot（Good Responders）`
- `y = diff_plot（Non-Good Responders）`

実行条件:
- 両群ともに有効データが **2点以上** 必要
- 条件を満たさない場合は `p = n/a`（NaN扱い）

### 4.4 ターミナル出力（統計結果）
`plot_good_vs_non_good.py` は実行時に以下をstdoutへ出力する。

- 対象人数:
  - `Total subjects`
  - `Good Responders`, `Non-Good Responders`
- 介入タイプ別内訳:
  - `Inc: Good=?, Non-Good=?`
  - `Dec: Good=?, Non-Good=?`
- 記述統計（各群）:
  - Mean / Median / SD / Range（min..max）
- Mann–Whitney U:
  - U statistic
  - p-value（小数4桁）
  - 星印（`***`, `**`, `*`, `n.s.`）

### 4.5 可視化（SVG）
出力:
- デフォルト: `Results/good_vs_non_good_comparison.svg`
- `--output` で変更可能

図の仕様（現行実装）:
- 箱ひげ図: seaborn `boxplot`（外れ値点は描かない `fliersize=0`）
- 個票点: seaborn `stripplot`（黒、jitterあり）
- 群:
  - `Non-Good Responders`
  - `Good Responders`
- 色:
  - Non-Good: `#AAAAAA`
  - Good: `#666666`
- y軸:
  - `diff_plot`（bpm）
  - 0の破線（`axhline(0)`）のみ表示
  - y範囲はデータ絶対値最大に基づき対称設定
- 有意性表示:
  - 上部に括弧（bracket）+ 星印（`p`から算出）
  - **数値のp値は図には載せない**（ターミナルに出力）

補足（図中のフッタ注釈）:
- 現行実装では `fig.text("** $\\it{p}$ < .01")` が固定文言として描画される。
  - bracketの星印は計算結果に依存するが、フッタは動的ではない点に注意。

CLI例:
- `python src/plot_good_vs_non_good.py --threshold 0.0`
- `python src/plot_good_vs_non_good.py --threshold 1.0 --output Results/good_vs_non_good_comparison.svg`

---

## 5. 主要ファイルのI/O仕様まとめ

### 5.1 生成物（Outputs）

| 種別 | パス | 生成元 | 内容 |
|---|---|---|---|
| 抽出CSV | `Data/Processed/{session}/extracted/{session}_ext.csv` | `ensure_extracted_csv()` | Raw txtから必要列抽出（追記） |
| セグメントCSV | `Data/Processed/{session}/split_segments/01_Feedback_Con_pre.csv` など | `split_feedback_segments()` | 60s + ±buffer のパディング、`_original_*`列付与 |
| 診断HTML | `Results/{session}/diagnosis_{segment}.html` | `process_segment()` | フィルタ・PSD・ピーク検出の診断 |
| ピークJSON | `Results/{session}/peaks_{segment}.json` | `process_segment()` | ピーク位置、timestamps、メタ情報 |
| Resting HR metrics | `Results/{session}/resting_hr_feedback_metrics.csv` | `run_session()` | セグメント毎のHRV/HR指標 + condition/phase 等 |
| 比較HTML | `Results/{session}/resting_hr_feedback_comparison.html` | `_plot_pre_post()` | Control/TargetのPre/Postプロット |
| PSDサマリHTML | `Results/{session}/hrv_psd_summary.html` | `run_session()` | AR(Burg) PSDの統合ページ |
| 欠損レポート | `Results/resting_hr_feedback_missing_segments.csv` | `--all-sessions` | セグメント欠損の一覧 |
| 条件別まとめHTML | `Results/resting_hr_feedback_comparison_Increase.html` 等 | `--all-sessions` | セッション比較HTMLの一覧 |
| Good responders CSV | `Results/good_responders.csv` 等 | `report_good_responders.py` | DID判定でGoodのみ抽出 |
| 群比較SVG | `Results/good_vs_non_good_comparison.svg` | `plot_good_vs_non_good.py` | Good vs Non-Goodの箱ひげ+点+星 |

---

## 6. 互換性・例外処理・注意点

### 6.1 `time_median_hr` の有無（後方互換）
- `run_resting_hr_feedback_analysis.py` の可視化、`report_good_responders.py`、`plot_good_vs_non_good.py` は
  `time_median_hr` を優先し、欠損時は `time_mean_hr` を使用する。

### 6.2 欠損セグメント
- `run_resting_hr_feedback_analysis.py --all-sessions` では欠損を許容し、レポートに残して継続する。
- 単一セッションモード（`--session`）では欠損は原則エラー（`allow_missing_segments=False`）。

### 6.3 統計検定の前提
- Mann–Whitney Uは2群の独立性を仮定する。
- 現行実装では1セッション=1被験者として扱う前提で、セッション間は独立としている。

---

## 付録A: 実行コマンド例（最小）

### A.1 セッション単体でResting HR Feedback解析
```bash
python src/run_resting_hr_feedback_analysis.py --session 251216_TK
```

### A.2 全セッション一括
```bash
python src/run_resting_hr_feedback_analysis.py --all-sessions
```

### A.3 Good Responders抽出
```bash
python src/report_good_responders.py --threshold 0.0 --output Results/good_responders_median.csv
```

### A.4 Good vs Non-Good 群比較（統計出力 + SVG生成）
```bash
python src/plot_good_vs_non_good.py --threshold 0.0 --output Results/good_vs_non_good_comparison.svg
```
