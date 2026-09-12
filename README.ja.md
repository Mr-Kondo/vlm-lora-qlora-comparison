# BASE vs LoRA vs QLoRA — 文書画像 → JSON

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr-Kondo/vlm-lora-qlora-comparison/blob/main/notebooks/vlm_lora_qlora_comparison.ipynb)

*English: [README.md](README.md)*

レシート画像を構造化 JSON に変換する視覚言語モデルのファインチューニングについて、3つの条件を統制比較します。

| 条件 | ベース重み | アダプタ | 学習対象パラメータ |
|---|---|---|---|
| **BASE** | 16-bit、未変更 | なし | 0 |
| **LoRA** | 16-bit | LoRA | アダプタのみ |
| **QLoRA** | 4-bit NF4（double quant） | 同等の LoRA | アダプタのみ |

この実験は品質だけを見るものではありません。次の4軸を同時に測定します。

**生成品質 × GPU VRAM × 学習時間 × 学習対象パラメータ数**

これにより、3条件の品質／資源トレードオフが暗示ではなく明示されます。

- **モデル:** [`HuggingFaceTB/SmolVLM-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)（2.2 B、Idefics3）
- **データセット:** [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) — レシート 800 訓練 / 100 検証 / 100 テスト
- **タスク:** 画像 → 正規化 JSON（`menu`、`sub_total`、`total`、`void_menu`）

---

## クイックスタート

### Google Colab

上のバッジを押してください。`notebooks/vlm_lora_qlora_comparison.ipynb` が Colab で開き、最初のセットアップセルが自分でこのリポジトリを `/content/vlm-lora-qlora-comparison` に clone します。アップロード作業は不要です。そのあと:

1. **ランタイム → ランタイムのタイプを変更 → T4 GPU**（以上）。GPU がないと学習も VRAM 測定もできず、最初のセルがその旨を表示します。
2. セルを上から順に実行します。ノートブックは下記の CLI を呼び出し、生成された成果物を読んで表とグラフを作ります。

`torch` と `torchvision` は意図的に `requirements.txt` に入れていません。Colab のものはランタイムの CUDA ビルドと整合しているためです。インストールセルの後に import が失敗する場合は **ランタイム → セッションを再起動** してから、インストールセルを飛ばしてバージョン確認セルから再実行してください。

**プリインストール済みパッケージを1つ削除する必要があります。** Colab は torch のビルドに紐づけて `torchao` を同梱していますが、peft はラップする全モジュールで torchao の有無を調べ、その判定関数が**バージョン不適合時に False を返さず例外を投げます**。そのため本プロジェクトが torchao を一切使わない（4-bit は bitsandbytes 経由）にもかかわらず、LoRA の注入が必ず失敗します。ノートブックには検出して削除するセルが入っています。ノートブック外では:

```bash
pip uninstall -y torchao
```

`train.py` と `evaluate.py` は数 GB のモデル読み込みより**前に**この状態を検査し、peft の内部で数分後に失敗するのではなく即座に修正方法を表示します。

**本番の前にスモークテストを通してください。** 学習セルに以下を追記すると数分で全体の流れを検証できます。通ったら外してください。

```
--set training.max_steps=4 --set data.max_train_samples=8 --set data.max_eval_samples=4
```

**所要時間の目安**（既定設定: 800 件 × 2 epoch）: L4/A100 で 2手法あわせて 1〜2 時間程度、T4 ではかなり長くなり 3〜4 時間を見込んでください。

**LoRA と QLoRA は同一セッションで学習してください。** 新しいセッションでは別の GPU が割り当てられることがあり、VRAM と学習時間の比較が無意味になります。どうしても分ける場合は、統制比較の出力（ノートブック section 7）の `same_gpu` 行を確認してください。これが失敗している場合、品質の比較は有効ですが**資源の比較は無効**です。

**切断に備える**には、学習前に出力先を Drive に向けてください。この1箇所の変更で、学習・評価・ノートブックの読み込み先がすべて揃います。

```python
from google.colab import drive; drive.mount('/content/drive')
!sed -i 's|output_root: outputs|output_root: /content/drive/MyDrive/vlm_ft_outputs|' configs/base.yaml
```

約 7 GB のモデル／データセットのダウンロードも Drive にキャッシュできます（読み込みが遅くなるトレードオフあり）。最初の `!python` セルより前に設定すれば、サブプロセスにも引き継がれます。

```python
import os; os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

### ローカル / 任意の CUDA ホスト

```bash
pip install -r requirements.txt

python scripts/download_model.py

python scripts/train.py --method lora  --config configs/lora.yaml
python scripts/train.py --method qlora --config configs/qlora.yaml

python scripts/evaluate.py --model-variant base  --config configs/base.yaml
python scripts/evaluate.py --model-variant lora  --config configs/lora.yaml
python scripts/evaluate.py --model-variant qlora --config configs/qlora.yaml

python scripts/collect_results.py
```

### 複数シードでの実験（推奨）

単一の実行ではファインチューニングが効くかは分かりますが、LoRA と QLoRA のわずかな差が本物かは判定できません。`scripts/run_experiment.py` は実験マトリクス全体を複数シードで実行し、2手法を**対応のある比較**（paired comparison）で評価します。両手法が同じシード列を使うため、シード毎に1つの差分が得られます。

```bash
python scripts/run_experiment.py --seeds 42 43 44 --dry-run   # 15 ステップを表示するだけ
python scripts/run_experiment.py --seeds 42 43 44 --smoke     # 配管の検証、数分
python scripts/run_experiment.py --seeds 42 43 44             # 本番
```

これはオーケストレーションだけを行います。上記と同じ `train.py` と `evaluate.py` を呼び出し、`<output_root>/seed<N>/` に書き出します。成果物が既に存在するステップはスキップするため、中断した実験（Colab のセッション回収など）は途中から再開できます。`--force` で全部やり直せます。1ステップが失敗しても残りは止まらず、最後に失敗一覧が表示され、その値は N/A として報告されます。

結果は `<output_root>/multiseed/` に、平均 ± 標本標準偏差とシード毎の生値として出力されます。p 値ではなく**符号の一致**を報告します。数シードしかない状況では「3シード中3シードで QLoRA が劣る」は解釈できますが、3点からの有意性検定は解釈できないためです。

本番実行前に数分で全体を通すには:

```bash
python scripts/train.py --method lora --set training.max_steps=4 --set data.max_train_samples=8
```

---

## 学習エントリポイントは1つ

`train_lora.py` / `train_qlora.py` は存在しません。`scripts/train.py --method {lora,qlora}` が、データセット読み込み・前処理・ターゲット整形・trainer ループ・チェックポイント処理・ロギング・メトリクス記録・シード処理・成果物レイアウトを共有します。手法が決めるのは次の2つだけです（`src/vlm_ft/modeling.py` の `METHOD_SPECS`）。

```python
lora  -> quantize_base=False, prepare_for_kbit=False   # ベースを 16-bit 計算 dtype で保持
qlora -> quantize_base=True,  prepare_for_kbit=True    # 同一ベースを 4-bit NF4 で読み、k-bit 学習用に準備
```

`configs/lora.yaml` と `configs/qlora.yaml` はどちらも `configs/base.yaml` を継承し、手法名と量子化ブロック以外は同一です。これは `diff configs/lora.yaml configs/qlora.yaml` で確認でき、`tests/test_config_and_methods.py` で強制されています。

同一に保たれるもの: モデル ID とリビジョン、データセットと3分割すべて、プロンプト、ターゲット JSON 形式、シード、エポック数、実効バッチサイズ、LoRA の rank / alpha / dropout / 対象モジュール、オプティマイザ、LR スケジュールと warmup、評価スケジュール、生成設定。

誤った比較を防ぐガードが2つあります。

- `--method lora` に `configs/qlora.yaml`（あるいは `load_in_4bit` の不一致）を渡すと、警告ではなくエラーになります。
- 各実行は、手法固有キーを除いた設定のフィンガープリントを記録します。`collect_results.py` が両者の一致を報告します。

### 除去できない差分

| パラメータ | LoRA | QLoRA | 理由 |
|---|---|---|---|
| ベース重みの保持形式 | 16-bit | 4-bit NF4 + double quant | これが独立変数 |
| `prepare_model_for_kbit_training` | 適用しない | 適用する | 量子化ベースには必須。非量子化では意味を持たない |
| 評価時の重み精度 | 16-bit | 4-bit（既定） | QLoRA の実運用形態。`--quantization none` で上書き可 |
| 評価損失の演算 | 16-bit forward | 4-bit forward | コードは同一、精度は同一でない |

同じ一覧と理由は、ノートブックにも表示され、すべての比較成果物に保存されます。

---

## 測定内容

### 品質 — `outputs/eval/<variant>/metrics.json`

`src/vlm_ft/metrics.py` が採点します。これは純 Python で、GPU 実行から独立に単体テストされています。比較の両側は同じ方法で正規化されます: 単一エントリの `menu` オブジェクトは1要素リストになり、キーはソートされ、値は空白を正規化した文字列になります。

| メトリクス | 定義 |
|---|---|
| JSON 妥当率 | 生成が JSON オブジェクトとしてパースできた割合（markdown フェンスや前後の散文は先に除去。全条件で同一処理） |
| 文書完全一致 | 正規化した予測文書が正規化した参照と一致 |
| フィールド正解率 | 正解の `key.path = value` のうち、リスト位置**込み**で完全に復元できた割合 |
| フィールド精度 / 再現率 / F1 | `key.path = value` のマイクロ平均。リスト位置を**含めない**ため、前の明細を取りこぼしてもずれた明細が正しければ正解として数えます |
| CER / WER | 正規化シリアライズ JSON 上の編集距離を文書横断でプール。パース不能な場合は生成文そのものにフォールバック |
| 評価損失 | ターゲットトークンのみのトークン重み付きクロスエントロピー（プロンプトと画像トークンはマスク） |

厳格版（位置考慮）の精度／再現率／F1 と、マクロ平均版も併せて記録されます。

分類精度と混同行列は**意図的に実装していません**。このタスクは可変形状の木構造と開語彙の文字列を出力するため、いずれにも意味のあるクラス集合が存在しません。

### VRAM — `outputs/<method>/resource_metrics.json`

測定区間の直前に `torch.cuda.reset_peak_memory_stats()` を呼び、終了後に `max_memory_allocated()` / `max_memory_reserved()` を読みます。両手法で同一の定義です。モデル読み込みと量子化は**別フェーズ**として計測するため、学習の数値に混入しません。CUDA デバイスがない場合は、推定せずに「測定不能」とその理由を記録します。

### 学習時間

`trainer.train()` を `time.perf_counter()` で挟み、両端で CUDA を同期します。ISO 形式の開始／終了時刻も記録します。モデルのダウンロード・読み込み・量子化は除外し、別途計測します。学習中の検証時間も計測し、合計に含めた値と除いた値の両方を報告します。`steps_per_second` / `samples_per_second` は派生的な追加情報で、合計時間の代替ではありません。

### パラメータ

設定からの推定ではなく、インスタンス化されたモデルから数えます。

```python
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

ただしここでは1つ重要な補正が必要です。bitsandbytes は NF4 の値を1バイトに2つ詰めるため、素朴な `numel()` は 4-bit モデルのパラメータ数を**半分**に報告します。`src/vlm_ft/resources.py` は `quant_state.shape` から論理的な個数を復元するため、QLoRA と LoRA は同一の総数を報告します。パラメータ**数**とパラメータ**メモリ使用量**は別フィールドに保存されます。量子化が変えるのは後者だけです。

BASE は定義上 `trainable = 0` を報告します。評価時はアダプタを凍結して読み込むため、素朴な `requires_grad` 集計ではファインチューニング済みモデルでも 0 になってしまいます。そのため、インスタンス化されたモデル上のアダプタテンソルを数えており、スモークテストがこれが学習時の報告値と一致することを検証しています。

---

## 成果物

```
outputs/
├── download_manifest.json          解決済みモデル SHA、ディスク使用量、分割サイズ
├── lora/
│   ├── adapter/                    adapter_config.json + 重み + プロセッサ
│   ├── resource_metrics.json       VRAM、時間、パラメータ、アダプタ設定、環境
│   ├── training_log_history.json   ステップ毎の訓練損失と評価毎の検証損失
│   └── run_config.json             完全解決済み設定 + CLI オーバーライド
├── qlora/                          同じレイアウト
├── eval/{base,lora,qlora}/
│   ├── metrics.json                全品質メトリクス + パラメータ + 推論設定
│   └── predictions.jsonl           文書毎の予測・参照・スコア
└── comparison/
    ├── comparison.json             統合表 + 分析 + 統制比較チェック
    ├── comparison.csv / .md
    └── loss_curves.json
```

複数シードの実験では、同じレイアウトが `<output_root>/seed<N>/` の下に入れ子になり、次が追加されます。

```
outputs/multiseed/
├── multiseed_comparison.json   メトリクス毎の平均/標準偏差/最小/最大、シード毎の対応差分、
│                               全シードに対する統制比較チェック
├── multiseed_summary.csv       統計量とシード毎の生値
├── multiseed_summary.md
└── multiseed_loss_curves.json
```

`resource_metrics.json` は次のスキーマに従います（追加キーは加算的）。

```json
{
  "method": "lora",
  "gpu": {"name": "...", "total_vram_bytes": 0,
          "peak_memory_allocated_bytes": 0, "peak_memory_reserved_bytes": 0},
  "training": {"duration_seconds": 0.0, "steps": 0, "samples": 0,
               "steps_per_second": null, "samples_per_second": null},
  "parameters": {"total": 0, "trainable": 0, "trainable_ratio": 0.0}
}
```

---

## 設定

ファイルを編集せずに任意の値を上書きできます。

```bash
python scripts/train.py --method lora \
  --set training.num_train_epochs=1 \
  --set data.max_train_samples=200 \
  --set data.image.longest_edge=512
```

知っておくとよい既定値:

| キー | 既定値 | 備考 |
|---|---|---|
| `data.image.longest_edge` | 768 | 2×2 タイル + 全体タイル1枚 = 画像トークン 405 個 |
| `data.max_seq_length` | 1536 | 実測: プロンプト 735 トークン、観測された最長ターゲット 480 — 切り詰めは発生しません |
| `training.gradient_accumulation_steps` | 8 | `per_device_train_batch_size: 1` で実効バッチサイズ 8 |
| `training.optim` | `adamw_torch` | 両手法で同一。アダプタのみ学習するため QLoRA に paged optimizer は不要 |
| `generation.max_new_tokens` | 512 | 観測された最長ターゲットをカバー |

切り詰めと教師信号のない例は collator が計数し、成果物に報告します。ターゲットの欠落が黙って起きることはありません。

### 解像度を上げる場合

`longest_edge` を上げると画像トークンが増えます（実測値、960×1280 の画像）。

| `longest_edge` | タイル数 | プロンプトトークン | 最長ターゲット(480)込み | 推奨 `max_seq_length` |
|---|---:|---:|---:|---:|
| 768（既定） | 5 | 735 | 1,217 | 1536 |
| 1152 | 10 | 1,191 | 1,673 | 2048 |
| 1536 | 13 | 1,465 | 1,947 | 2048 |
| 1920 | 21 | 2,194 | 2,676 | 3072 |

解像度は**必ず `configs/base.yaml` で変更してください**。コマンド毎の `--set` では、5つのコマンドのどれかで指定を忘れると前処理が揃わず比較が壊れます。base.yaml を変えれば全条件に伝播します。

---

## ハードウェアと実行時間

比較が意味を持つには、両手法が GPU に載る必要があります。既定設定ではモデルは 2.2 B パラメータ、系列長は約 1,200 トークン、gradient checkpointing は有効です。

**既定値は A100 / L4 クラスの GPU を想定しています。** `per_device_train_batch_size: 4` と `gradient_accumulation_steps: 2`（実効バッチサイズは 8 のままで、batch 1 × 8 と同一）により、batch 1 では遊んでしまうアクセラレータを使い切ります。

| GPU | 備考 |
|---|---|
| **A100 40 GB** | 推奨。bf16 が使え、メモリ帯域 1,555 GB/s。この処理は演算律速ではなく**帯域律速**なので、ピーク TFLOPS より帯域が効きます |
| **L4 22.5 GB** | 問題なし。bf16 は使えますが帯域は約 300 GB/s なので明確に遅くなります |
| **T4 16 GB** | 動きますが bf16 非対応（fp16 にフォールバック）かつ低速。`--set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8` を**両方の実行に**追加してください |

この既定値では VRAM は制約になりません。L4 でも余裕があります。A100 を選ぶ理由は容量ではなく**速度**です。

- L4/A100 で2手法あわせて 1〜2 時間程度、T4 ではかなり長くなります。
- CUDA OOM が出たら `data.image.longest_edge` と `data.max_seq_length` を下げてください。同じ変更を**両方の実行に**適用すること。
- `training.gradient_checkpointing: true` は維持してください。無効にすると速くなりますが、活性値メモリがピークを支配するようになり、**QLoRA の VRAM 削減が実際より小さく見えます**。これは測定対象そのものです。無効にする場合は両手法で無効にし、その旨を明記してください。

---

## テスト

```bash
pip install -r requirements-dev.txt
pytest -q
```

カバー範囲: メトリクス定義（Levenshtein 高速経路を純 Python 参照実装とランダム照合するテストを含む）、設定の継承と手法ガード、レポート／表の組み立て（すべての N/A 経路を含む）、ノートブックの構造、そして小型スタンドインモデルに対する `train.py` と `evaluate.py` の CPU エンドツーエンド実行。

### 検証済みのこと、未検証のこと

この開発環境（macOS、CPU のみ）で検証済み: LoRA の学習パス全体、両方の評価パス、すべての成果物スキーマ、メトリクス実装、ノートブックの分析・描画セル。`hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration` をスタンドインとして**実データの CORD** に対してエンドツーエンドで実行し、系列長の予算は実物の SmolVLM プロセッサで実測しました。

**未実行:** CUDA と bitsandbytes を要する QLoRA 分岐、および GPU を要する実際の VRAM / 学習時間の測定。QLoRA の設定経路（NF4 + double quant + 計算 dtype 一致の `BitsAndBytesConfig`）は単体テスト済みで、4-bit パラメータ計数の補正も実装済みですが、最初の QLoRA 実行は短いスモークテスト（`--set training.max_steps=4`）から始めてください。

検証済み依存バージョン: transformers 5.17.0、peft 0.20.0、datasets 5.0.1、accelerate 1.15.0、torch 2.14.0。transformers 4.46+ 向けの互換シムも備えています（`dtype` と `torch_dtype`、`AutoModelForImageTextToText` と `AutoModelForVision2Seq`、および transformers 5 で削除された `warmup_ratio` の明示的なステップ数変換）。

---

## 構成

```
configs/       base.yaml と、それを継承する2つの手法設定
src/vlm_ft/    config, data, metrics, modeling, resources, report, seeding
scripts/       download_model.py, train.py, evaluate.py, run_experiment.py, collect_results.py
tests/         メトリクス・設定・レポート・ノートブック・CPU エンドツーエンドのテスト
notebooks/     Colab 比較ノートブック
```
