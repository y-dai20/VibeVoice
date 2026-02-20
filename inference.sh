
#!/bin/bash

uv run python demo/vibevoice_asr_inference_from_file.py \
    --model_path microsoft/VibeVoice-ASR \
    --audio_files /workspace/VibeVoice/finetuning-asr/test.webm
