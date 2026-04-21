# VibeVoice

このディレクトリでは、VibeVoice ASR のローカル実験用メモとして、以下をまとめています。

- **ベースモデル推論**
- **LoRA ファインチューニング**
- **ファインチューニング済みモデルでの推論**
- **バッチ推論・簡易評価**

## セットアップ

```bash
uv sync
```

W&B を使う場合は事前にログインしておきます。

```bash
uv run wandb login
```

事前に、ファインチューニング済みモデルのウェイトを Wasabi から落としておきます。

```bash
mkdir -p models
aws s3 sync \
  s3://titan-research-and-development/vibevoice-asr/output_20260408_051424_best/ \
  models/output_20260408_051424_best/
```

Wasabi 用の `aws` CLI 設定や認証情報は事前に通してある前提です。以降、学習済み LoRA を使うときは例えば次のようにローカル展開先を `--lora_path` に指定します。

```bash
--lora_path models/output_20260408_051424_best
```

`finetuning-asr/train.sh` や `finetuning-asr/inference.sh` では `HF_HUB_ENABLE_HF_TRANSFER=0` と `HF_HUB_DISABLE_XET=1` を付けています。Hugging Face 周りで転送まわりの問題を避けたい場合は、同じ環境変数をそのまま使う運用で大丈夫です。

## ベースモデルで推論する

`inference.py` がベースモデルの推論エントリです。

### 1 ファイルだけ推論

```bash
uv run python inference.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_files /path/to/sample.mp3 \
  --batch_size 1 \
  --num_beams 1 \
  --max_new_tokens 8192 \
  --output_base_dir outputs
```

主な出力は `outputs/<timestamp>/` 配下に保存されます。

- **`inference.json`**: 推論結果全体
- **`inference.rttm`**: 話者区間
- **`run_args.json`**: 実行引数のスナップショット

### ディレクトリ単位で推論

```bash
uv run python inference.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_dir /path/to/audio_dir \
  --batch_size 2 \
  --num_beams 3 \
  --content_no_repeat_ngram_size 3
```

対応する入力は `wav`, `mp3`, `flac`, `mp4`, `m4a`, `webm` です。

### シェルスクリプトを使う

`inference.sh` が雛形です。

```bash
bash inference.sh
```

ただし、スクリプト中の `--audio_files` の値は環境依存のサンプルになっているため、実際に使う前に対象ファイルまたは対象ディレクトリへ書き換えて使う前提です。

## LoRA ファインチューニング

学習エントリは `finetuning-asr/lora_finetune.py` です。

作業ディレクトリ:

```bash
cd finetuning-asr
```

### 学習データ形式

JSON と音声ファイルを同じディレクトリに置きます。

```text
dataset_dir/
├── 0.mp3
├── 0.json
├── 1.mp3
├── 1.json
└── ...
```

JSON の基本形:

```json
{
  "audio_duration": 351.73,
  "audio_path": "0.mp3",
  "segments": [
    {
      "speaker": 0,
      "text": "Hey everyone, welcome back...",
      "start": 0.0,
      "end": 38.68
    },
    {
      "speaker": 1,
      "text": "Thanks for having me...",
      "start": 38.75,
      "end": 77.88
    }
  ],
  "customized_context": ["domain term 1", "domain term 2"]
}
```

`customized_context` は任意ですが、固有名詞や業務用語を補助情報として入れたいときに使えます。

### まずは `train.sh` を使う

`train.sh` は現在の実験設定をそのまま再現するための入口です。

```bash
bash train.sh
```

このスクリプトでは以下のような設定で学習します。

- **ベースモデル**: `microsoft/VibeVoice-ASR`
- **トークナイザ**: `Qwen/Qwen2.5-1.5B`
- **出力先**: `./output_YYYYMMDD_HHMMSS`
- **LoRA**: `r=8`, `alpha=16`, `dropout=0.05`
- **学習率**: `7e-5`
- **validation**: `--validation_split_ratio 0.05`
- **定期テスト推論**: `--test_data_dir` に評価用ディレクトリを指定
- **繰り返し抑制**: `--content_no_repeat_ngram_size 3`

`train.sh` の `--data_dir` や `--test_data_dir` はサンプル値なので、自分の環境に合わせて書き換えて使います。

### 直接コマンドで学習する

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
uv run torchrun --nproc_per_node=1 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --tokenizer_path Qwen/Qwen2.5-1.5B \
  --data_dir /path/to/train_dataset \
  --output_dir ./output \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 7e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 20 \
  --validation_split_ratio 0.05 \
  --content_no_repeat_ngram_size 3 \
  --gradient_checkpointing \
  --bf16 \
  --report_to wandb
```

### 学習中・学習後の成果物

- **`output_xxx/checkpoint-<step>/`**: 中間 checkpoint
- **`output_xxx/`**: 最終 LoRA adapter, processor, metrics

推論時の `--lora_path` には、最終出力ディレクトリか、特定の `checkpoint-*` を指定できます。途中 checkpoint を比較したいときは `checkpoint-*` を使うのが便利です。

## ファインチューニング済みモデルで推論する

単発推論は `inference_lora.py` を使います。

### まずは `inference.sh`

```bash
bash inference.sh
```

中身は概ね次のような実行です。

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
uv run python inference_lora.py \
  --base_model microsoft/VibeVoice-ASR \
  --lora_path /path/to/output_or_checkpoint \
  --audio_file /path/to/audio.mp3 \
  --context_info "" \
  --max_new_tokens 16384 \
  --num_beams 3 \
  --content_no_repeat_ngram_size 3 \
  --output_json lora_inference.json \
  --output_rttm lora_inference.rttm
```

出力:

- **`lora_inference.json`**: raw text と structured segments
- **`lora_inference.rttm`**: RTTM 形式の話者区間

## ファインチューニング済みモデルでディレクトリ一括推論する

複数音声をまとめて処理する場合は `batch_inference_lora.py` を使います。

### まずは `batch_inference.sh`

```bash
bash batch_inference.sh
```

このスクリプトでは以下のようなことができます。

- **`--audio_dir`**: 入力ディレクトリ指定
- **`--output_dir`**: JSON 出力先
- **`--skip_existing`**: 既存結果をスキップ
- **`--num_samples`**: ランダムに件数を絞る
- **`--min_duration` / `--max_duration`**: 音声長でフィルタ

直接実行例:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
uv run python batch_inference_lora.py \
  --base_model microsoft/VibeVoice-ASR \
  --lora_path /path/to/output_or_checkpoint \
  --audio_dir /path/to/audio_dir \
  --output_dir ./batch_results \
  --num_beams 3 \
  --content_no_repeat_ngram_size 3 \
  --skip_existing
```

各音声ごとに 1 JSON が保存されます。

## テストデータで簡易評価する

`test.sh` は `test_lora.py` を呼び出して、LoRA モデルを入力ディレクトリに対して評価・出力するためのショートカットです。

```bash
bash test.sh
```

実験時に「ある checkpoint を assets に対してまとめて流して比較したい」用途で使えます。

## よく使う引数

- **`--num_beams`**: `1` なら greedy/sampling、`2` 以上で beam search
- **`--temperature`**: `0` なら deterministic に近い推論
- **`--max_new_tokens`**: 長尺音声では大きめが必要
- **`--content_no_repeat_ngram_size`**: 自作のカスタムロジットプロセッサーで n-gram の繰り返しを制限するための値。Qwen2.5 でデコード時に直前の文字と同じ文字を延々と繰り返して JSON 形式が壊れることがあるため、その崩れを抑えて適切な JSON フォーマットへ寄せる目的で使う
- **`--validation_split_ratio`**: 学習データから validation を自動分割
- **`--test_data_dir`**: 学習中に定期的な test 推論を回したいときに使う

## 運用メモ

- **パスはそのままでは使えないものがある**
  `train.sh`, `inference.sh`, `batch_inference.sh`, `test.sh` の一部は環境依存のサンプルパスを含むので、実行前に必ず自分の環境へ合わせて更新します。

- **`lora_path` は final output でも checkpoint でもよい**
  比較用途なら `checkpoint-20`, `checkpoint-40` のような途中重み指定が便利です。

- **長尺音声は `max_new_tokens` と repetition 制御が重要**
  長い音源で崩れやすい場合は、`--num_beams 3` と `--content_no_repeat_ngram_size 3` 前後から試すと扱いやすいです。特に `content_no_repeat_ngram_size` は、Qwen2.5 のデコード時に同じ文字列を繰り返して JSON 構造が崩れるケースを矯正するために入れています。
  `finetuning-asr/README.md` に LoRA 学習のより詳しいオプション説明があります。
