
#!/bin/bash

uv run python inference.py \
    --model_path microsoft/VibeVoice-ASR \
    --num_beams 1 \
    --content_no_repeat_ngram_size 10 \
    --content_no_repeat_decode_max_tokens 1024 \
    --content_no_repeat_debug \
    --audio_files /home/yamad/workspace/tests/assets/youtube/イクサガミ_600.mp4
