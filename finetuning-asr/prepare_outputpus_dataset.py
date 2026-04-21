#!/usr/bin/env python3
"""Prepare a toy_dataset-style directory from speechlab outputs.

This script recursively finds source JSON files under the source tree.

Supported inputs:

- ``dataset_segments.json`` produced by speechlab
- ``Vibe-Voice/**/inference.json`` produced by VibeVoice

Examples:

- ``<episode>/dataset_segments.json``
- ``<episode>/chunks/chunk_000/dataset_segments.json``
- ``<episode>/Vibe-Voice/<timestamp>/inference.json``
- ``<episode>/chunks/chunk_000/Vibe-Voice/<timestamp>/inference.json``

Each sample is flattened into the output directory as:

- ``<sample_id>.json``
- ``<sample_id>.<audio_ext>``
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy speechlab/VibeVoice JSON outputs and their source audio into "
            "a flat dataset directory compatible with finetuning-asr."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/workspace/speechlab/outputs"),
        help="Directory containing speechlab output subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/VibeVoice/finetuning-asr/outputs"),
        help="Destination directory for flattened dataset files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the destination directory.",
    )
    parser.add_argument(
        "--input-format",
        choices=["dataset_segments", "vibevoice", "auto"],
        default="dataset_segments",
        help=(
            "Input JSON format to collect. "
            "'dataset_segments' reads speechlab outputs, "
            "'vibevoice' reads VibeVoice inference.json, "
            "and 'auto' reads both."
        ),
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        default=0,
        help="Skip samples with fewer than this many segments.",
    )
    parser.add_argument(
        "--min-mix-speakers",
        type=int,
        default=0,
        help="Skip samples with fewer than this many unique speakers.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Skip samples shorter than this duration in seconds.",
    )
    parser.add_argument(
        "--japanese-only",
        action="store_true",
        help="Keep only samples whose segment texts look predominantly Japanese.",
    )
    return parser.parse_args()


def iter_dataset_files(source_dir: Path, input_format: str) -> Iterable[Path]:
    if input_format == "dataset_segments":
        return sorted(source_dir.rglob("dataset_segments.json"))
    if input_format == "vibevoice":
        return sorted(
            path
            for path in source_dir.rglob("inference.json")
            if "Vibe-Voice" in path.parts
        )

    candidates = {
        *source_dir.rglob("dataset_segments.json"),
        *(
            path
            for path in source_dir.rglob("inference.json")
            if "Vibe-Voice" in path.parts
        ),
    }
    return sorted(candidates)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_sample_id(json_path: Path, source_dir: Path) -> str:
    full_parts = json_path.parts

    if "chunks" in full_parts:
        chunks_index = full_parts.index("chunks")
        if chunks_index + 1 < len(full_parts):
            chunk_name = full_parts[chunks_index + 1]
            if chunk_name.startswith("chunk_"):
                if chunks_index >= 1:
                    episode_id = full_parts[chunks_index - 1]
                    return f"{episode_id}_{chunk_name}"
                return chunk_name

    if "Vibe-Voice" in full_parts:
        vibe_index = full_parts.index("Vibe-Voice")
        if vibe_index >= 1:
            return full_parts[vibe_index - 1]

    relative_parts = json_path.relative_to(source_dir).parts
    return relative_parts[0]


def resolve_audio_source(payload: Dict[str, Any], json_path: Path) -> Path:
    metadata = payload.get("metadata") or {}
    candidates = [
        metadata.get("speech_input_path"),
        metadata.get("input_path"),
        payload.get("audio_path"),
        payload.get("audio_file"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (json_path.parent / candidate_path).resolve()
        if candidate_path.exists():
            return candidate_path
    # Fallback for chunk-level outputs when metadata is incomplete.
    chunk_dir = json_path.parent
    if chunk_dir.name.startswith("chunk_") and chunk_dir.parent.name == "chunks":
        chunk_audio_dir = chunk_dir.parent / "audio"
        for extension in (".mp3", ".wav", ".flac", ".m4a"):
            candidate_path = chunk_audio_dir / f"{chunk_dir.name}{extension}"
            if candidate_path.exists():
                return candidate_path

    raise FileNotFoundError(f"Audio source not found for {json_path}")


def infer_audio_duration(payload: Dict[str, Any], audio_src: Path | None = None) -> float:
    if audio_src is not None:
        segments = normalize_segments(payload, audio_src)
        if not segments:
            return 0.0
        return float(max(segment.get("end", 0.0) for segment in segments))

    explicit = payload.get("audio_duration")
    if explicit is not None:
        return float(explicit)

    metadata = payload.get("metadata") or {}
    if metadata.get("audio_duration") is not None:
        return float(metadata["audio_duration"])

    segments = canonical_segments(payload)
    if not segments:
        return 0.0
    return float(max(segment.get("end", 0.0) for segment in segments))


def is_chunk_audio(payload: Dict[str, Any], audio_src: Path) -> bool:
    metadata = payload.get("metadata") or {}
    speech_input_path = metadata.get("speech_input_path")
    if speech_input_path:
        try:
            if Path(speech_input_path).name == audio_src.name and "chunks/audio" in speech_input_path:
                return True
        except TypeError:
            pass
    return audio_src.stem.startswith("chunk_")


def normalize_segments(
    payload: Dict[str, Any], audio_src: Path
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = canonical_segments(payload)
    metadata = payload.get("metadata") or {}
    audio_offset = float(metadata.get("audio_offset") or 0.0)

    if not segments:
        return []

    if not is_chunk_audio(payload, audio_src) or audio_offset <= 0:
        return segments

    normalized: List[Dict[str, Any]] = []
    for segment in segments:
        normalized_segment = dict(segment)
        if "start" in normalized_segment:
            normalized_segment["start"] = round(
                float(normalized_segment["start"]) - audio_offset, 6
            )
        if "end" in normalized_segment:
            normalized_segment["end"] = round(
                float(normalized_segment["end"]) - audio_offset, 6
            )
        normalized.append(normalized_segment)
    return normalized


def build_label_payload(
    payload: Dict[str, Any], audio_src: Path, audio_filename: str
) -> Dict[str, Any]:
    segments = normalize_segments(payload, audio_src)
    label = {
        "audio_duration": (
            float(max(segment.get("end", 0.0) for segment in segments))
            if segments
            else 0.0
        ),
        "audio_path": audio_filename,
        "segments": segments,
    }
    if "customized_context" in payload:
        label["customized_context"] = payload["customized_context"]
    elif payload.get("context_info") is not None:
        label["customized_context"] = payload["context_info"]
    return label


def count_segments(payload: Dict[str, Any]) -> int:
    metadata = payload.get("metadata") or {}
    segment_count = metadata.get("segment_count")
    if segment_count is not None:
        try:
            return int(segment_count)
        except (TypeError, ValueError):
            pass
    return len(canonical_segments(payload))


def count_unique_speakers(payload: Dict[str, Any]) -> int:
    segments: List[Dict[str, Any]] = canonical_segments(payload)
    speakers = {
        str(segment["speaker"])
        for segment in segments
        if segment.get("speaker") not in (None, "")
    }
    return len(speakers)


def speaker_label_from_id(value: Any) -> str:
    text = str(value)
    if text.startswith("speaker_"):
        return text
    return f"speaker_{text}"


def speaker_id_as_int(value: Any) -> int | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.startswith("speaker_"):
        text = text[len("speaker_") :]

    try:
        return int(text)
    except (TypeError, ValueError):
        pass

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    return int(number)


def extract_vibevoice_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = payload.get("segments")
    if isinstance(segments, list):
        return segments

    results = payload.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and isinstance(item.get("segments"), list):
                return item["segments"]
    return []


def canonical_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_segments: List[Dict[str, Any]] = []
    segments = payload.get("segments")
    if isinstance(segments, list):
        raw_segments = segments
    else:
        raw_segments = extract_vibevoice_segments(payload)

    if not raw_segments:
        return []

    first = raw_segments[0]
    if {"start", "end", "speaker"}.issubset(first):
        normalized_existing: List[Dict[str, Any]] = []
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            if "start" not in segment or "end" not in segment:
                continue
            speaker_id = speaker_id_as_int(segment.get("speaker"))
            if speaker_id is None:
                continue
            normalized_existing.append(
                {
                    "start": round(float(segment["start"]), 6),
                    "end": round(float(segment["end"]), 6),
                    "speaker": speaker_id,
                    "text": str(segment.get("text", "")).strip(),
                }
            )
        return normalized_existing

    normalized: List[Dict[str, Any]] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        if "start_time" not in segment or "end_time" not in segment:
            continue
        if "speaker_id" not in segment:
            continue
        speaker_id = speaker_id_as_int(segment.get("speaker_id"))
        if speaker_id is None:
            continue
        normalized.append(
            {
                "start": round(float(segment["start_time"]), 6),
                "end": round(float(segment["end_time"]), 6),
                "speaker": speaker_id,
                "text": str(segment.get("text", "")).strip(),
            }
        )
    return normalized


KANA_CHAR_RE = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
DEFAULT_MIN_JAPANESE_RATIO = 0.8
DEFAULT_JAPANESE_SEGMENT_CONFIDENCE_THRESHOLD = 0.8


def japanese_text_ratio(text: str) -> float:
    if not text:
        return 0.0

    kana_chars = len(KANA_CHAR_RE.findall(text))
    cjk_chars = len(CJK_CHAR_RE.findall(text))
    latin_chars = len(LATIN_CHAR_RE.findall(text))
    relevant_chars = kana_chars + cjk_chars + latin_chars

    if relevant_chars == 0:
        return 0.0

    if kana_chars == 0:
        return (0.15 * cjk_chars) / relevant_chars

    weighted_japanese_chars = kana_chars + (0.5 * cjk_chars)
    confidence = (weighted_japanese_chars / relevant_chars) + 0.2
    return min(confidence, 1.0)


def sample_japanese_ratio(payload: Dict[str, Any]) -> float:
    segments: List[Dict[str, Any]] = canonical_segments(payload)
    segment_texts = [
        str(segment.get("text", "")).strip() for segment in segments
    ]
    segment_texts = [text for text in segment_texts if text]
    if not segment_texts:
        return 0.0

    passed_segments = sum(
        1
        for text in segment_texts
        if japanese_text_ratio(text) >= DEFAULT_JAPANESE_SEGMENT_CONFIDENCE_THRESHOLD
    )
    return passed_segments / len(segment_texts)


def should_keep_sample(
    payload: Dict[str, Any],
    audio_src: Path,
    min_segments: int,
    min_mix_speakers: int,
    min_duration: float,
    japanese_only: bool,
) -> bool:
    if count_segments(payload) < min_segments:
        return False
    if count_unique_speakers(payload) < min_mix_speakers:
        return False
    if infer_audio_duration(payload, audio_src=audio_src) < min_duration:
        return False
    if japanese_only and sample_japanese_ratio(payload) < DEFAULT_MIN_JAPANESE_RATIO:
        return False
    return True


def copy_sample(
    json_path: Path,
    payload: Dict[str, Any],
    audio_src: Path,
    source_dir: Path,
    output_dir: Path,
    force: bool,
) -> str:
    sample_id = build_sample_id(json_path, source_dir)
    audio_dst = output_dir / f"{sample_id}{audio_src.suffix.lower()}"
    json_dst = output_dir / f"{sample_id}.json"

    if not force:
        conflicts = [path for path in (audio_dst, json_dst) if path.exists()]
        if conflicts:
            names = ", ".join(path.name for path in conflicts)
            raise FileExistsError(
                f"Destination file already exists for sample {sample_id}: {names}"
            )

    shutil.copy2(audio_src, audio_dst)
    label_payload = build_label_payload(payload, audio_src, audio_dst.name)
    json_dst.write_text(
        json.dumps(label_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sample_id


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    dataset_files = list(iter_dataset_files(source_dir, args.input_format))
    if not dataset_files:
        raise FileNotFoundError(
            f"No matching input JSON files found under {source_dir} "
            f"(input_format={args.input_format})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    copied_ids: List[str] = []
    skipped_ids: List[str] = []
    failed_samples: List[str] = []
    for json_path in dataset_files:
        sample_id = build_sample_id(json_path, source_dir)
        payload = load_json(json_path)
        try:
            audio_src = resolve_audio_source(payload, json_path)
            if not should_keep_sample(
                payload,
                audio_src=audio_src,
                min_segments=args.min_segments,
                min_mix_speakers=args.min_mix_speakers,
                min_duration=args.min_duration,
                japanese_only=args.japanese_only,
            ):
                skipped_ids.append(sample_id)
                continue
            copied_ids.append(
                copy_sample(
                    json_path,
                    payload,
                    audio_src,
                    source_dir,
                    output_dir,
                    force=args.force,
                )
            )
        except Exception as exc:
            failed_samples.append(sample_id)
            print(f"Skipping failed sample {sample_id}: {exc}")
            continue

    print(f"Prepared {len(copied_ids)} samples in {output_dir}")
    for sample_id in copied_ids:
        print(f"- {sample_id}")
    if skipped_ids:
        print(
            f"Skipped {len(skipped_ids)} samples by filters "
            f"(min_segments={args.min_segments}, "
            f"min_mix_speakers={args.min_mix_speakers}, "
            f"min_duration={args.min_duration}, "
            f"japanese_only={args.japanese_only}, "
            f"min_japanese_ratio={DEFAULT_MIN_JAPANESE_RATIO}, "
            f"japanese_segment_confidence_threshold="
            f"{DEFAULT_JAPANESE_SEGMENT_CONFIDENCE_THRESHOLD})"
        )
    if failed_samples:
        print(f"Skipped {len(failed_samples)} samples due to errors")
        for sample_id in failed_samples:
            print(f"- {sample_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
