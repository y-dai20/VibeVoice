#!/bin/bash

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run python inference_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path ./output \
    --audio_file ./test.webm \
    --context_info "" \
    --output_json lora_inference.json \
    --output_rttm lora_inference.rttm
