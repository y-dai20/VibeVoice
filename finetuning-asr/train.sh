#!/bin/bash

set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="./output_${timestamp}"
echo "Training output_dir: ${output_dir}"

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run torchrun --nproc_per_node=1 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --tokenizer_path Qwen/Qwen2.5-1.5B \
  --data_dir ./toy_dataset \
  --test_data_dir ./toy_dataset \
  --output_dir "${output_dir}" \
  --lora_r 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.0 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --save_steps 10 \
  --gradient_checkpointing \
  --bf16 \
  --report_to none
