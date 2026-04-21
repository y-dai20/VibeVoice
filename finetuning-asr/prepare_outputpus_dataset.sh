

uv run python prepare_outputpus_dataset.py \
  --source-dir /workspace/speechlab/dataset/outputs_v2 \
  --output-dir ./dataset \
  --min-segments 10 \
  --min-mix-speakers 2 \
  --japanese-only
