## Usage

### Resting HR (Feedback) pre/post comparison

Run the end-to-end pipeline (Raw → extract → 4 Feedback segments → diagnosis/R-peaks → HRV → plot):

- Single session:
	- `python src/run_resting_hr_feedback_analysis.py --session 251216_TK`
- Rebuild extracted CSV (use if Raw files changed):
	- `python src/run_resting_hr_feedback_analysis.py --session 251216_TK --rebuild-extracted`
- All sessions under `Data/Raw`:
	- `python src/run_resting_hr_feedback_analysis.py --all-sessions`

Outputs are written under:
- `Data/Processed/{session}/extracted/` and `Data/Processed/{session}/split_segments/`
- `Results/{session}/` (HTML diagnosis reports, peaks JSON, HRV CSV, comparison plot)
# ECG Analysis Pipeline

## 必要環境

- Python 3.8+（動作確認は 3.10+ 推奨）
- 依存ライブラリ: [requirements.txt](requirements.txt)

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ディレクトリ構成（重要）

このプロジェクトの「解析入力」は、セッションごとに分割済みのセグメントCSVです。

```
Data/
	Raw/                       # 元データ（例: Cognionicsの .txt）
	Processed/
		{session}/
			split_segments/        # 解析対象（Step 1が読む）
				01_*.csv
				02_*.csv
	Trigger/                   # トリガー/アノテーション関連（任意）

Results/
	{session}/                 # Step 1/2 の出力先
```

## 入力データ（セグメントCSV）の形式

- 必須: ECG列（列名に `ExGa` を含むものを自動検出）
	- 例: `ExGa 1(uV)`
- 推奨: `Time (s)`（ない場合はサンプリングレートから生成）
- 推奨: `Timestamp`（前処理・分割時の参照用。Step 1/2 では必須ではありません）

単位変換について:

- ECG列名に `uV` が含まれる場合、Step 1 で µV→mV（`/1000`）へ自動変換します。

サンプリングレート:

- デフォルト 500 Hz（[src/analysis_pipeline/config.py](src/analysis_pipeline/config.py)）

## 実行手順（現行パイプライン）

### Step 1: 信号診断 + Rピーク検出（HTML + JSON）

セッション内の全セグメントを処理:

```bash
python src/run_signal_diagnosis.py --session 251216_TK
```

特定セグメントのみ処理（セグメント名 = CSVファイル名（拡張子なし））:

```bash
python src/run_signal_diagnosis.py --session 251216_TK --segment 03_Resting_HR_1_Set1
```

利用可能セッションの一覧（`Data/Processed/{session}/split_segments/*.csv` が存在するもの）:

```bash
python src/run_signal_diagnosis.py --list-sessions
```

### Step 2: HRV指標算出（CSV）

Step 1 で生成された `peaks_*.json` を入力として、HRVを集計します。

```bash
python src/run_hrv_metrics.py --session 251216_TK
```

特定セグメントのみ:

```bash
python src/run_hrv_metrics.py --session 251216_TK --segment 03_Resting_HR_1_Set1
```

### Step 3: 群レベル統計 + 可視化（Protocol v2）

Step 2 で生成された HRV集計CSV（`Results/hrv_summaries/*_hrv_summary.csv`）を入力として、
条件情報（`Data/conditions.csv`）を付与し、群レベルの統計検定と論文掲載用の図を出力します。

初回実行時に `Data/conditions.csv` が存在しない場合は、
`Results/hrv_summaries` 内に存在する全セッションIDを網羅したテンプレートを自動生成し、
「記入後に再実行してください」というメッセージを出して正常終了します。

`Data/conditions.csv` の必須カラム:

- `session_id`: セッション名（例: `251216_TK`）
- `group`: 被験者グループ（`Increase` または `Decrease`）
- `Set1_Cond`: Set1の介入条件（`Increase` / `Decrease` / `Control`）
- `Set2_Cond`: Set2の介入条件（`Increase` / `Decrease` / `Control`）

実行例:

```bash
# 初回: conditions.csv を自動生成（記入してから再実行）
python src/run_group_stats.py

# 2回目以降: 統計 + 図を出力
python src/run_group_stats.py
```

出力先:

- `Results/stats/`

### 追加: 心拍フィードバック（Session）前半/後半の効果確認（デフォルト各3分）

`split_segments/` 内のファイル名に `Session` を含むセグメント（複数可）を対象に、
前半 `3分` と後半 `3分` のHR/HRV指標を算出して差分（後半−前半）を出力します。

```bash
# 全セッションを対象（デフォルト: 180秒=3分）
python src/run_hr_feedback_effect.py

# 特定セッションのみ
python src/run_hr_feedback_effect.py --session 251216_TK

# ウィンドウ長変更（例: 120秒=2分）
python src/run_hr_feedback_effect.py --window-sec 120
```

出力:

- `Results/stats/hr_feedback_effect_180s.csv`（window秒数に応じてファイル名が変わります）

CSVの見方（効果を見る基本）:

- 1行 = 1つの `Session*` セグメント（例: `04_Session`）
- `early_*` が前半、`late_*` が後半
- `diff_*` は「後半 − 前半」
- `pct_*` は「(後半−前半)/前半 × 100」

まず見る列（おすすめ）:

- `diff_time_mean_hr`（bpm）: 平均心拍の変化
- `diff_time_rmssd`（ms）: 副交感指標の変化（増えると“ゆらぎ”増）
- `diff_freq_lf_hf_ratio`: 自律神経バランスの変化（解釈は研究方針に合わせて）

品質チェック（最低限）:

- `early_removal_rate` / `late_removal_rate`: RR除外率が高い行は信頼性に注意
- `early_rpeak_quality_score` / `late_rpeak_quality_score`: Rピーク検出品質の目安
- `notes` / `early_quality_notes` / `late_quality_notes`: 窓が短い・重なった等の警告

## 出力

出力は次の場所に保存されます。

- `Results/{session}/diagnosis_{segment}.html`
	- セグメントごとの診断レポート（極性、フィルタ前後波形、PSD、検出ピークなど）
- `Results/{session}/peaks_{segment}.json`
	- 検出ピーク（インデックス・時刻）とメタ情報（反転、skewness、品質ノート等）
- `Results/hrv_summaries/{session}_hrv_summary.csv`
	- セッション内セグメントのHRV集計（SDNN, RMSSD, pNN50, LF/HF 等）
- `Results/stats/`
	- 群レベル統計の結果CSVと、論文掲載用の図（Step 3）

## 前処理ユーティリティ（必要な場合のみ）

データの整形・アノテーションからの分割などは、`src/` 直下のスクリプト群で行います。
研究データの状況に応じて使い分けてください。

- [src/extract_ecg_columns.py](src/extract_ecg_columns.py)
	- Cognionicsのテキストから解析に必要な列を抽出してCSV化
- [src/detect_trigger_changes.py](src/detect_trigger_changes.py)
	- `TRIGGER(DIGITAL)` の変化点を検出してTimestamp一覧を出力
- [src/insert_virtual_triggers.py](src/insert_virtual_triggers.py)
	- 別セッション等のトリガー列を基準点からオフセットコピーして補完
- [src/deduplicate_triggers.py](src/deduplicate_triggers.py)
	- 連続重複トリガー削除
- [src/add_annotation.py](src/add_annotation.py)
	- 既存CSVに `Annotation` 列を追記
- [src/split_by_annotation.py](src/split_by_annotation.py)
	- アノテーションペアに基づき `split_segments/` を生成

## プロジェクト構成（現状）

```
ECG_Analize/
	src/
		analysis_pipeline/
			config.py
			io_utils.py
			preprocess.py
			diagnosis.py
			rpeak.py
			hrv.py
		run_signal_diagnosis.py
		run_hrv_metrics.py
		extract_ecg_columns.py
		split_by_annotation.py
		detect_trigger_changes.py
		insert_virtual_triggers.py
		deduplicate_triggers.py
		add_annotation.py
		visualize_ecg.py
	Data/
	Results/
	References/
	requirements.txt
	README.md
```

## 参考

- [References/Analysis.R](References/Analysis.R) にR実装の処理が残っています（ピーク検出の整合などの参照用）。
