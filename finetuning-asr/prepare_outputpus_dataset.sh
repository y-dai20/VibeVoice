

uv run python prepare_outputpus_dataset.py \
  --source-dir /workspace/speechlab/dataset/japanese_tver_drama_outputs_v2 \
  --output-dir ./japanese_tver_drama_outputs_v2 \
  --min-segments 10 \
  --min-mix-speakers 2 \
  --min-duration 500
