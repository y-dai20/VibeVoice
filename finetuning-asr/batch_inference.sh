#!/bin/bash

# Batch inference script for processing multiple audio files from a directory
# Usage: ./batch_inference.sh

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run python batch_inference_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path /workspace/VibeVoice/finetuning-asr/output_20260310_090824/checkpoint-40 \
    --audio_dir /workspace/dataset/wasabi/apple_podcast_dataset/video \
    --output_dir /workspace/VibeVoice/finetuning-asr/apple_podcast_dataset \
    --context_info "" \
    --max_new_tokens 16384 \
    --num_beams 3 \
    --content_no_repeat_ngram_size 3 \
    --skip_existing \
    --num_samples 20 \
    --min_duration 600 \
    --max_duration 1800
