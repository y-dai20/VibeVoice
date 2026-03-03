#!/bin/bash

HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 uv run python test_lora.py \
    --base_model microsoft/VibeVoice-ASR \
    --lora_path /home/yamad/workspace/tests/speechlab/asr/vibevoice/VibeVoice/finetuning-asr/output_20260302_043041/checkpoint-40 \
    --input_dir /home/yamad/workspace/tests/speechlab/assets \
    --num_beams 3 \
    --output_dir ./batch_results
