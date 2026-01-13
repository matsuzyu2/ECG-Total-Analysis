# 心拍FB中の心拍変動解析（During HRFB Analysis）仕様書

作成日: 2026-01-13

---

## A. 概要（Overview）

### 目的
本機能は、**心拍フィードバック（HRFB）中の心拍変化**に注目し、各HRFBセッションにおける

- **開始直後5分（First5: 0–5分）**
- **終了直前5分（Last5: 10–15分）**

の2区間を抽出して **平均心拍（Mean HR）** を算出し、
**Delta = Last5 − First5** として「開始から終了に向けて心拍がどう変化したか」を比較・可視化します。

### 関連する主要スクリプト
- `src/run_during_hrfb_analysis.py`
  - HRFB中のFirst5/Last5セグメントの切り出し → Rピーク検出 → 指標計算 → セッション別CSV出力
- `src/plot_during_hrfb_delta.py`
  - セッション別CSVを集約し、Deltaを計算して Box + Swarm の静的SVGを出力

---

## B. データフローと依存関係（Data Flow）

### 入力データ

#### 1) Timestamp_2.csv
- パス: `Data/Timestamp/Timestamp_2.csv`
- 使用カラム:
  - `Date`（YYYY_MM_DD）
  - `Subject`（例: `TK`, `MS_1` 等。SubjectData側の表記に一致する必要がある）
  - `HRFB_1`, `HRFB_2`（HHMM形式の開始時刻）

利用方法:
- `run_during_hrfb_analysis.py` は、対象セッションの `Date` と `Subject` を特定した後、
  同一行の `HRFB_1` / `HRFB_2` を **HRFB開始時刻（HHMM）** として利用します。
- 値は文字列で格納されていてもよく（例: `"""1438"""` 等）、引用符は除去されます。
- 欠損・不正（数字でない等）の場合、そのセグメントは **行を残したままNaNとして記録**します（後述）。

#### 2) SubjectData.csv
- パス: `Data/Subject/SubjectData.csv`
- 使用カラム:
  - `Date`（YYYY_MM_DD）
  - `Subject`（例: `TK`, `MS_1` 等）
  - `Order`（1/2）: HRFB_1/HRFB_2 が Target/Control のどちらかを決める
  - `BF_Type`（`Inc` / `Dec`）: グルーピング（増加群/減少群）に使用

Target/Control 判定ロジック（Orderによる）:
- `Order = 1` の場合
  - `HRFB_1` → `Target`
  - `HRFB_2` → `Control`
- `Order = 2` の場合
  - `HRFB_1` → `Control`
  - `HRFB_2` → `Target`
- `Order` が取得できない場合は `Unknown` とします（ただしプロット側は `Target` / `Control` を前提に参照します）。

#### 3) Raw ECG data
- パス: `Data/Raw/<session_id>/*.txt`
- `run_during_hrfb_analysis.py` は、必要に応じて raw `.txt` から解析用の抽出CSV（`*_ext.csv`）を生成します。

### 処理の流れ（高レベル）

1. 生データ（Raw `.txt`）
2. 抽出CSV生成（`extract_ecg_columns.py`）
3. HRFB開始時刻（HHMM）にもとづくセグメント切り出し（`split_by_feedback_timestamp.py` 拡張API）
4. セグメントの診断・Rピーク検出（`run_signal_diagnosis.process_segment`）
   - 診断HTMLとピークJSONを生成
5. ピークJSONから指標算出（`run_hrv_metrics.process_peaks_file`）
6. セッション別の結果CSV出力（`Results/<session_id>/during_hrfb_metrics.csv`）
7. 複数セッション集約 → Delta計算 → 可視化（`plot_during_hrfb_delta.py` → SVG出力）

---

## C. 実装詳細（Implementation Details）

### 1. split_by_feedback_timestamp.py（拡張部分）

#### 新規追加API: `split_segment_by_start_and_duration`
目的:
- **任意の開始HHMM + 継続時間**でECGを切り出すための汎用API。
- During HRFB（First5/Last5）のように、既存の「Feedback_*」列名に依存しないウィンドウ定義に対応します。

関数シグネチャ（概略）:
- `split_segment_by_start_and_duration(session_id, extracted_csv_path, segment_key, start_hhmm, duration_sec, start_offset_sec=0, buffer_sec=15, chunksize=..., allow_missing=False) -> Path | None`

入力/挙動:
- `start_hhmm`:
  - HHMM（例: `1436`）または引用符付き文字列などを許容し、内部で正規化します。
- `start_offset_sec`:
  - HHMMに対してさらに秒オフセットを加算します（例: Last5用に +600秒）。
- `duration_sec`:
  - 元の計測区間（paddingを除いた区間）の長さ（During HRFBでは 300秒）。
- `buffer_sec`:
  - フィルタ端のアーティファクト対策として、抽出時に前後へ付与するpadding（秒）。
  - `run_during_hrfb_analysis.py` 側からは `Config.FILTER_PADDING_SEC` を渡します。

出力:
- セグメントCSV:
  - `Data/Processed/<session_id>/split_segments/<segment_key>.csv`
- セグメントCSVには以下のメタ情報列を付与します:
  - `_original_start_ts`: paddingを除いた本来の開始日時
  - `_original_end_ts`: paddingを除いた本来の終了日時

欠損時の扱い:
- `allow_missing=True` の場合、指定範囲に該当データが無ければ `None` を返します（空ファイルは残しません）。
- `allow_missing=False` の場合は例外を送出します。

#### なぜ既存関数（`split_feedback_segments`）を使わず、このAPIを追加したか
- 既存の `split_feedback_segments` は、Timestamp側の列名探索が **"Feedback" を含む列**を前提としており、
  `HRFB_1` / `HRFB_2` のような列には対応しません。
- During HRFBは「2つの開始時刻（HRFB_1/2）× 相対オフセット（0分/10分）」という定義のため、
  **列名依存を排除して再利用可能なAPI**を追加するのが最も堅牢です。

---

### 2. run_during_hrfb_analysis.py（解析処理）

#### セグメント定義（First5 / Last5）
各セッションにつき、以下4セグメントを処理します（合計4行を出力する設計）。

- HRFB_1:
  - `during_HRFB1_First5`: `start_offset_sec = 0`, `duration_sec = 300`
  - `during_HRFB1_Last5`:  `start_offset_sec = 600`, `duration_sec = 300`
- HRFB_2:
  - `during_HRFB2_First5`: `start_offset_sec = 0`, `duration_sec = 300`
  - `during_HRFB2_Last5`:  `start_offset_sec = 600`, `duration_sec = 300`

padding（buffer）:
- 切り出し時に `Config.FILTER_PADDING_SEC` 秒のpaddingを付けて抽出します。
- 下流の `process_segment` がpaddingを考慮してフィルタ後にトリミングします。

#### 条件判定ロジック

1) Subject名の解決（resolve_session_identity）
- `session_id`（例: `251216_TK`）から日付とSubjectトークンを取り出し、
  `SubjectData.csv` の `Date` 行を候補に絞り込みます。
- `Subject` が完全一致しない場合、プレフィックス一致（例: `MS` → `MS_1`）で一意に決めます。

2) Orderによる Target/Control 割当
- `Order` は `SubjectData.csv` から読み込みます。
- `condition` は以下の規則で付与します:
  - `Order=1`: HRFB_1→Target, HRFB_2→Control
  - `Order=2`: HRFB_1→Control, HRFB_2→Target
  - それ以外/取得不可: `Unknown`

注意:
- `segment_key`（例: `during_HRFB1_First5`）は物理的識別子であり、Target/Controlは含めません。
  解析結果CSV内の `condition` 列が意味ラベル（Target/Control）を担います。

#### 指標計算とフォールバック

処理手順（各セグメント）:
1. `split_segment_by_start_and_duration(..., allow_missing=True)` でCSVを切り出し
2. `run_signal_diagnosis.process_segment(segment_csv, session_id, config=..., verbose=...)`
   - Rピーク検出と品質評価、診断HTML/ピークJSONの生成
3. `run_hrv_metrics.process_peaks_file(peaks_path, ...)` でHRV指標を計算
   - `metrics.to_flat_dict()` によりフラットな辞書へ展開

`time_mean_hr` の取得優先順位（フォールバック）:
- `time_mean_hr`（最優先）
- `mean_hr`
- `meanHR`
- `hr_mean`
- いずれも無い/変換できない場合は `NaN`

#### エラーハンドリング（NaN行を残す仕様）

設計方針:
- 欠損・失敗があっても **行を除外しない**。
- 後段の集計/可視化で「欠損」か「処理失敗」かを区別できるよう、
  `status` と `error_msg` を必ず記録します。

主な `status` 値:
- `ok`: 指標計算まで完了
- `missing_timestamp`: Timestamp_2.csv で開始時刻が欠損/不正
- `split_failed`: セグメント切り出し処理が例外で失敗
- `missing_segment`: 切り出しは試みたが該当行が0件（allow_missing=TrueでNone）
- `diagnosis_failed`: `process_segment` が失敗（例外 or 戻り値False）
- `metrics_failed`: `process_peaks_file` が失敗

出力行数:
- 原則として各セッション **4行**（HRFB_1/2 × First5/Last5）を出力することを意図しています。
  （欠損でも行は残り、`time_mean_hr` は NaN になります）

---

### 3. plot_during_hrfb_delta.py（可視化）

#### Deltaの定義
- Delta はセッション内・条件別に以下で定義します:
  - `Delta = MeanHR(Last5) − MeanHR(First5)`
- MeanHR は `during_hrfb_metrics.csv` の `time_mean_hr` を使用します。

#### 入力と集約
- 入力: `Results/<session_id>/during_hrfb_metrics.csv`
- 必須列: `condition`, `phase`, `time_mean_hr`, `BF_Type`, `Subject`
- 各セッションから以下2値を算出:
  - `control_delta`（ControlのLast5−First5）
  - `target_delta`（TargetのLast5−First5）
- `BF_Type` が `Dec` または `Inc` のセッションのみプロット対象（それ以外は除外）

#### プロット仕様

X軸カテゴリ（並び順は固定）:
1. `Con (Dec)`
2. `Dec`
3. `Con (Inc)`
4. `Inc`

表示ラベル:
- `Con (Dec)` → `Control`
- `Dec` → `Dec`
- `Con (Inc)` → `Control`
- `Inc` → `Inc`

描画スタイル:
- Box plot + Swarm plot（カテゴリごとにスウォーム点を重ねる）
- 各カテゴリに平均±SE（標準誤差）をエラーバーで重ね描き
- y軸は対称レンジ（データ最大絶対値 × 1.15、ただし最低±1.0）
- yラベル: `Δ HR (bpm): Last5 − First5`

#### 出力形式
- 静的SVGとして保存（HTMLは出力しない）

---

## D. 出力ファイル仕様（Output Specifications）

### 中間/セッション別出力

#### 1) セッション別メトリクスCSV
- パス: `Results/<session_id>/during_hrfb_metrics.csv`

必須列（必ず存在するよう補完されます）:
- `session_id`: セッションID（例: `251216_TK`）
- `Date`: YYYY_MM_DD
- `Subject`: SubjectData上のSubject（例: `TK`, `MS_1`）
- `BF_Type`: `Inc` / `Dec`
- `Order`: 1/2（取得不能時は空/NaN）
- `hrfb_index`: 1 または 2
- `condition`: `Target` / `Control` / `Unknown`
- `phase`: `First5` / `Last5`
- `time_mean_hr`: 平均心拍（bpm）。欠損/失敗時はNaN
- `status`: 処理状態（ok / missing_timestamp / ...）
- `error_msg`: 失敗理由（空文字の場合あり）
- `quality_notes`: 品質メモ（計算できた場合に限り値が入ることがある）
- `segment_key`: 例 `during_HRFB1_First5`

補足:
- `metrics.to_flat_dict()` の内容が追加列としてCSVに含まれる場合があります（HRV指標など）。

#### 2) セッション別診断HTML / ピークJSON（副産物）
`process_segment` により、以下がセッションResults配下に生成されます（ファイル名はstemに依存）。
- `Results/<session_id>/diagnosis_<segment_key>.html`
- `Results/<session_id>/peaks_<segment_key>.json`

### 最終成果物（集約可視化）

#### 1) DeltaサマリSVG
- パス: `Results/during_hrfb_delta_summary.svg`
- 内容:
  - BF_Type（Dec/Inc）× condition（Control/Target）を組み合わせた4カテゴリのDelta分布
  - Box plot + Swarm + 平均±SE

---

## 実行方法（参考）

解析（全セッション）:
- `python src/run_during_hrfb_analysis.py`

解析（単一セッション）:
- `python src/run_during_hrfb_analysis.py --session 251216_TK`

可視化（SVG）:
- `python src/plot_during_hrfb_delta.py --results-dir Results --output Results/during_hrfb_delta_summary.svg`
