#!/bin/bash

set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="./output_${timestamp}"
echo "Training output_dir: ${output_dir}"

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run torchrun --nproc_per_node=1 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --tokenizer_path Qwen/Qwen2.5-1.5B \
  --data_dir ./dataset_v2 \
  --lora_r 4 \
  --lora_alpha 8 \
  --output_dir "${output_dir}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-5 \
  --bf16 \
  --report_to none
