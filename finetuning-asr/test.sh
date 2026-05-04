#!/bin/bash

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run python test_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path /workspace/VibeVoice/finetuning-asr/output_20260417_074455_l4_lr7e5_b8_1307_69_v2/checkpoint-280 \
    --input_dir /workspace/assets \
    --num_beams 3 \
    --content_no_repeat_ngram_size 3 \
    --output_dir ./batch_results
