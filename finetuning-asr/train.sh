#!/bin/bash

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run torchrun --nproc_per_node=1 lora_finetune.py \
  --model_path microsoft/VibeVoice-ASR \
  --tokenizer_path Qwen/Qwen2.5-1.5B \
  --data_dir ./dataset \
  --output_dir ./output \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --bf16 \
  --report_to none
