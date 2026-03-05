#!/bin/bash

uv run python quantize_vibevoice.py \
  --output_dir ./vibevoice-asr-4bit \
  --quantization 4bit
