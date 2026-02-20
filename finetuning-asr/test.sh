#!/bin/bash

uv run python inference_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path ./output \
    --audio_file ./test.wav \
    --context_info ""
