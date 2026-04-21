#!/bin/bash

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run python inference_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path /workspace/VibeVoice/finetuning-asr/output_20260310_090824/checkpoint-40 \
    --audio_file /workspace/dataset/wasabi/apple_podcast_dataset/video/1000078829789.mp3 \
    --context_info "" \
    --max_new_tokens 16384 \
    --num_beams 3 \
    --content_no_repeat_ngram_size 3 \
    --output_json lora_inference.json \
    --output_rttm lora_inference.rttm
