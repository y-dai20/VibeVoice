
#!/bin/bash

uv run python demo/vibevoice_asr_inference_from_file.py \
    --model_path microsoft/VibeVoice-ASR \
    --output_json inference.json \
    --output_rttm inference.rttm \
    --num_beams 3 \
    --audio_files ./test.webm
