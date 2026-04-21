#!/bin/bash

set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="./output_${timestamp}"
echo "Training output_dir: ${output_dir}"

export WANDB_PROJECT="${WANDB_PROJECT:-vibevoice-asr-lora-2}"
export WANDB_WATCH="${WANDB_WATCH:-false}"

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run torchrun --nproc_per_node=1 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --tokenizer_path Qwen/Qwen2.5-1.5B \
  --data_dir /workspace/dataset_v2 \
  --test_data_dir /workspace/assets \
  --output_dir "${output_dir}" \
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
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --save_steps 20 \
  --validation_split_ratio 0.05 \
  --content_no_repeat_ngram_size 3 \
  --gradient_checkpointing \
  --bf16 \
  --report_to wandb
