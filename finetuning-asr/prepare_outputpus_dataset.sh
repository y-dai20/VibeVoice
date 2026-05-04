
# Apple podcast
uv run python prepare_outputpus_dataset.py \
  --source-dir /workspace/speechlab/dataset/outputs_v2 \
  --output-dir /workspace/pseudo_v2 \
  --report-path /workspace/speechlab/dataset/outputs_v2/report.csv \
  --min-segments 20 \
  --min-mix-speakers 3 \
  --max-der 0.5

# Apple podcast 2
uv run python prepare_outputpus_dataset.py \
  --source-dir /workspace/speechlab/dataset/apple_podcast_outputs_v2 \
  --output-dir /workspace/pseudo_v2 \
  --report-path /workspace/speechlab/dataset/apple_podcast_outputs_v2/report.csv \
  --min-segments 20 \
  --min-mix-speakers 3 \
  --max-der 0.5

# TVer Variety
uv run python prepare_outputpus_dataset.py \
  --source-dir /workspace/speechlab/dataset/japanese_tver_variety_outputs_v2 \
  --output-dir /workspace/pseudo_v2 \
  --report-path /workspace/speechlab/dataset/japanese_tver_variety_outputs_v2/report.csv \
  --min-segments 20 \
  --max-confusion 100 \
  --min-mix-speakers 3

# TVer drama
uv run python prepare_outputpus_dataset.py \
  --source-dir /workspace/speechlab/dataset/japanese_tver_drama_outputs_v2 \
  --output-dir /workspace/pseudo_v2 \
  --report-path /workspace/speechlab/dataset/japanese_tver_drama_outputs_v2/report.csv \
  --min-segments 20 \
  --max-confusion 100 \
  --min-mix-speakers 4
