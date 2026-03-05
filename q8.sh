#!/bin/bash

uv run python quantize_vibevoice.py \
  --output_dir ./vibevoice-asr-8bit \
  --quantization 8bit
