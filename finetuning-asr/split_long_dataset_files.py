#!/usr/bin/env python3
"""Split long diarization dataset audio/JSON pairs into two halves."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split diarization dataset audio/JSON pairs into _part1/_part2 files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing <stem>.mp3 and <stem>.json pairs.",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=None,
        help="Base filenames to split, without extension. "
        "If omitted, --max-duration must be set and all qualifying files are split automatically.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Automatically split every JSON whose audio_duration exceeds this value (seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for split files. Defaults to --dataset-dir.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Directory to move originals into. Defaults to <dataset-dir>/original_long.",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Do not move original files after splitting.",
    )
    parser.add_argument(
        "--audio-extension",
        default=".mp3",
        help="Audio extension to look for and emit. Default: .mp3",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_ffmpeg(
    input_audio: Path, output_audio: Path, start_sec: float, duration: float
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_audio),
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(output_audio),
    ]
    subprocess.run(command, check=True)


def split_segments(
    segments: List[Dict[str, Any]], split_point: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    part1: list[dict[str, Any]] = []
    part2: list[dict[str, Any]] = []

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])

        if end <= split_point:
            clipped = dict(segment)
            clipped["start"] = round(start, 2)
            clipped["end"] = round(end, 2)
            part1.append(clipped)
            continue

        if start >= split_point:
            shifted = dict(segment)
            shifted["start"] = round(start - split_point, 2)
            shifted["end"] = round(end - split_point, 2)
            part2.append(shifted)
            continue

        left_overlap = split_point - start
        right_overlap = end - split_point
        if left_overlap >= right_overlap:
            clipped = dict(segment)
            clipped["start"] = round(start, 2)
            clipped["end"] = round(split_point, 2)
            part1.append(clipped)
        else:
            shifted = dict(segment)
            shifted["start"] = 0.0
            shifted["end"] = round(end - split_point, 2)
            part2.append(shifted)

    return part1, part2


def split_pair(
    dataset_dir: Path,
    output_dir: Path,
    backup_dir: Path,
    stem: str,
    audio_extension: str,
    keep_originals: bool,
) -> None:
    json_path = dataset_dir / f"{stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")

    payload = load_json(json_path)
    # Prefer the extension recorded in the JSON's audio_path over the CLI default
    recorded_ext = Path(payload.get("audio_path", "")).suffix
    if recorded_ext:
        audio_extension = recorded_ext
    audio_path = dataset_dir / f"{stem}{audio_extension}"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    duration = float(payload["audio_duration"])
    split_point = duration / 2.0
    part1_segments, part2_segments = split_segments(
        payload.get("segments", []), split_point
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    part_specs = [
        ("part1", 0.0, split_point, part1_segments),
        ("part2", split_point, duration - split_point, part2_segments),
    ]

    for suffix, start_sec, part_duration, segments in part_specs:
        output_audio = output_dir / f"{stem}_{suffix}{audio_extension}"
        output_json = output_dir / f"{stem}_{suffix}.json"
        run_ffmpeg(audio_path, output_audio, start_sec, part_duration)

        part_payload = dict(payload)
        part_payload["audio_path"] = output_audio.name
        part_payload["audio_duration"] = round(part_duration, 2)
        part_payload["segments"] = segments
        save_json(output_json, part_payload)

    if not keep_originals:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(audio_path), str(backup_dir / audio_path.name))
        shutil.move(str(json_path), str(backup_dir / json_path.name))

    print(
        f"{stem}: split={split_point:.3f}s, "
        f"part1_segments={len(part1_segments)}, part2_segments={len(part2_segments)}"
    )


def find_long_stems(
    dataset_dir: Path, max_duration: float, audio_extension: str
) -> List[str]:
    """Return stems of JSON files whose audio_duration exceeds max_duration."""
    stems = []
    for json_path in sorted(dataset_dir.glob("*.json")):
        stem = json_path.stem
        # Skip files that are already split parts
        if stem.endswith("_part1") or stem.endswith("_part2"):
            continue
        try:
            payload = load_json(json_path)
        except Exception:
            continue
        if float(payload.get("audio_duration", 0)) > max_duration:
            stems.append(stem)
    return stems


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or dataset_dir).resolve()
    backup_dir = (args.backup_dir or (dataset_dir / "original_long")).resolve()
    audio_extension = args.audio_extension
    if not audio_extension.startswith("."):
        audio_extension = f".{audio_extension}"

    if args.stems:
        stems = args.stems
    elif args.max_duration is not None:
        stems = find_long_stems(dataset_dir, args.max_duration, audio_extension)
        print(f"Found {len(stems)} file(s) exceeding {args.max_duration}s: {stems}")
    else:
        raise SystemExit("Either --stems or --max-duration must be specified.")

    for stem in stems:
        split_pair(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            backup_dir=backup_dir,
            stem=stem,
            audio_extension=audio_extension,
            keep_originals=args.keep_originals,
        )


if __name__ == "__main__":
    main()
