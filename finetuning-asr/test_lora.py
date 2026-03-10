#!/usr/bin/env python
"""
Inference with a LoRA fine-tuned VibeVoice ASR model.

This script supports:
  - single-file inference
  - batch inference over a directory
  - optional RTTM export
  - optional DER evaluation with pyannote.metrics

Examples:
    python test_lora.py \
        --base_model microsoft/VibeVoice-ASR \
        --lora_path ./output \
        --audio_file ./toy_dataset/0.mp3

    python test_lora.py \
        --base_model microsoft/VibeVoice-ASR \
        --lora_path ./output \
        --input_dir ./toy_dataset \
        --output_dir ./batch_results
"""

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from vibevoice.generation_mixin import ContentNoRepeatGenerationMixin
from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _parse_time_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except ValueError:
        return None
    return None


def _rttm_file_id(name: str) -> str:
    base = Path(str(name)).stem or "audio"
    cleaned = re.sub(r"\s+", "_", base)
    cleaned = re.sub(r"[^\w.\-]+", "_", cleaned, flags=re.UNICODE)
    return cleaned or "audio"


def _segment_to_fields(
    seg: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    speaker_value = seg.get("speaker_id", seg.get("speaker"))
    start = _parse_time_seconds(seg.get("start_time", seg.get("start")))
    end = _parse_time_seconds(seg.get("end_time", seg.get("end")))
    if speaker_value is None:
        speaker = None
    elif str(speaker_value).startswith("speaker_"):
        speaker = str(speaker_value)
    else:
        speaker = f"speaker_{speaker_value}"
    return speaker, start, end


def _segments_to_rttm_lines(
    segments: List[Dict[str, Any]],
    file_id: str,
) -> List[str]:
    lines: List[str] = []
    for seg in segments or []:
        speaker, start, end = _segment_to_fields(seg)
        if speaker is None or start is None or end is None:
            continue
        duration = end - start
        if duration <= 0:
            continue
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return lines


def _default_rttm_path(
    output_json: Optional[str], output_rttm: Optional[str]
) -> Optional[Path]:
    if output_rttm:
        return Path(output_rttm)
    if output_json:
        return Path(output_json).with_suffix(".rttm")
    return None


def _annotation_from_rttm_lines(lines: Iterable[str], uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    for line_no, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        if duration <= 0:
            continue
        track = f"{speaker}@{line_no}"
        annotation[Segment(start, start + duration), track] = speaker
    return annotation


def _load_rttm_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"RTTM file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _load_reference_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Reference JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_segments_from_json(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Reference JSON must contain a 'segments' list.")
    return segments


def _context_info_from_json(payload: Dict[str, Any]) -> Optional[str]:
    customized_context = payload.get("customized_context")
    if isinstance(customized_context, list):
        values = [str(item).strip() for item in customized_context if str(item).strip()]
        if values:
            return "\n".join(values)
    if isinstance(customized_context, str) and customized_context.strip():
        return customized_context.strip()
    return None


def _calculate_der(
    reference_lines: List[str],
    hypothesis_lines: List[str],
    uri: str,
    collar: float,
    skip_overlap: bool,
) -> Dict[str, Any]:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    reference = _annotation_from_rttm_lines(reference_lines, uri)
    hypothesis = _annotation_from_rttm_lines(hypothesis_lines, uri)
    details = metric(reference, hypothesis, detailed=True)
    der = metric.compute_metric(details)
    mapping = metric.optimal_mapping(reference, hypothesis)
    return {
        "der": der,
        "der_percent": der * 100.0,
        "details": {key: float(value) for key, value in dict(details).items()},
        "ref_speakers": sorted(map(str, reference.labels())),
        "hyp_speakers": sorted(map(str, hypothesis.labels())),
        "optimal_mapping": {str(k): str(v) for k, v in mapping.items()},
    }


def _summarize_aggregate_der(
    metric: DiarizationErrorRate,
    evaluated_files: int,
    collar: float,
    skip_overlap: bool,
) -> Optional[Dict[str, Any]]:
    if evaluated_files == 0:
        return None
    details = metric[:]
    der = metric.compute_metric(details)
    return {
        "evaluated_files": evaluated_files,
        "collar": collar,
        "skip_overlap": skip_overlap,
        "der": der,
        "der_percent": der * 100.0,
        "details": {key: float(value) for key, value in dict(details).items()},
    }


def _resolve_reference_path(
    audio_path: Path,
    explicit_path: Optional[str],
    explicit_dir: Optional[str],
    suffix: str,
    batch_root: Optional[Path],
) -> Optional[Path]:
    if explicit_path:
        return Path(explicit_path)

    if explicit_dir:
        root = Path(explicit_dir)
        if batch_root is not None:
            try:
                relative = audio_path.relative_to(batch_root).with_suffix(suffix)
                candidate = root / relative
                if candidate.exists():
                    return candidate
            except ValueError:
                pass
        candidate = root / audio_path.with_suffix(suffix).name
        if candidate.exists():
            return candidate

    sidecar = audio_path.with_suffix(suffix)
    if sidecar.exists():
        return sidecar
    return None


def load_lora_model(
    base_model_path: str,
    lora_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    print(f"Loading base model from {base_model_path}")

    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model_path, language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )

    if device != "auto":
        model = model.to(device)

    print(f"Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("Model loaded successfully")

    return model, processor


def transcribe(
    model,
    processor,
    audio_path: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    num_beams: int = 1,
    context_info: str = None,
    device: str = "cuda",
    seed: Optional[int] = None,
    content_no_repeat_ngram_size: int = 0,
    content_no_repeat_decode_max_tokens: int = 2048,
    content_no_repeat_debug: bool = False,
):
    print(f"\nTranscribing: {audio_path}")

    inputs = processor(
        audio=audio_path,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "num_beams": num_beams,
        "do_sample": False if num_beams > 1 else (temperature > 0),
    }
    if gen_config["do_sample"]:
        gen_config["temperature"] = temperature
        gen_config["top_p"] = 0.9
    logits_processor = (
        ContentNoRepeatGenerationMixin.build_content_no_repeat_logits_processor(
            tokenizer=processor.tokenizer,
            content_no_repeat_ngram_size=content_no_repeat_ngram_size,
            content_no_repeat_decode_max_tokens=content_no_repeat_decode_max_tokens,
            content_no_repeat_debug=content_no_repeat_debug,
        )
    )

    if seed is not None:
        set_global_seed(seed)

    with torch.no_grad():
        if logits_processor is not None:
            output_ids = model.generate(
                **inputs,
                **gen_config,
                logits_processor=logits_processor,
            )
        else:
            output_ids = model.generate(**inputs, **gen_config)

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    cleaned_text = generated_text.strip()
    if cleaned_text.lower().startswith("assistant"):
        cleaned_text = cleaned_text.split("\n", 1)[-1].strip()

    try:
        segments = processor.post_process_transcription(cleaned_text)
    except Exception as exc:
        print(f"Warning: Failed to parse structured output: {exc}")
        segments = []

    return {
        "raw_text": generated_text,
        "segments": segments,
    }


def _build_input_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if args.audio_file:
        items.append({"audio_path": Path(args.audio_file).expanduser().resolve()})
        return items

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        if args.audio_glob is not None and not path.match(args.audio_glob):
            continue
        items.append({"audio_path": path.resolve()})

    if not items:
        raise FileNotFoundError(
            f"No supported media files were found directly under {input_dir}"
        )

    if args.limit is not None:
        items = items[: args.limit]
    return items


def _prepare_item_metadata(
    item: Dict[str, Any],
    args: argparse.Namespace,
    batch_root: Optional[Path],
) -> Dict[str, Any]:
    audio_path: Path = item["audio_path"]
    file_id = _rttm_file_id(str(audio_path))

    reference_json_path = _resolve_reference_path(
        audio_path=audio_path,
        explicit_path=args.reference_json if args.audio_file else None,
        explicit_dir=args.reference_json_dir,
        suffix=".json",
        batch_root=batch_root,
    )
    reference_rttm_path = _resolve_reference_path(
        audio_path=audio_path,
        explicit_path=args.reference_rttm if args.audio_file else None,
        explicit_dir=args.reference_rttm_dir,
        suffix=".rttm",
        batch_root=batch_root,
    )

    context_info = args.context_info
    reference_json_payload = None
    if reference_json_path is not None and reference_json_path.exists():
        reference_json_payload = _load_reference_json(reference_json_path)
        if context_info is None:
            context_info = _context_info_from_json(reference_json_payload)

    metadata = {
        "audio_path": audio_path,
        "file_id": file_id,
        "context_info": context_info,
        "reference_json_path": reference_json_path,
        "reference_rttm_path": reference_rttm_path,
        "reference_json_payload": reference_json_payload,
    }
    if batch_root is not None:
        metadata["relative_path"] = str(audio_path.relative_to(batch_root))
    else:
        metadata["relative_path"] = audio_path.name
    return metadata


def _build_reference_rttm_lines(
    item: Dict[str, Any],
) -> Tuple[Optional[List[str]], Optional[str]]:
    reference_rttm_path = item["reference_rttm_path"]
    if reference_rttm_path is not None and Path(reference_rttm_path).exists():
        return _load_rttm_lines(Path(reference_rttm_path)), str(
            Path(reference_rttm_path).resolve()
        )

    payload = item.get("reference_json_payload")
    if payload is not None:
        segments = _reference_segments_from_json(payload)
        return _segments_to_rttm_lines(segments, item["file_id"]), str(
            Path(item["reference_json_path"]).resolve()
        )

    return None, None


def _print_single_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Transcription Result")
    print("=" * 60)

    print("\n--- Raw Output ---")
    raw_text = result["raw_text"]
    print(raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text)

    if result["segments"]:
        print(f"\n--- Structured Output ({len(result['segments'])} segments) ---")
        for seg in result["segments"][:20]:
            print(
                f"[{seg.get('start_time', 'N/A')} - {seg.get('end_time', 'N/A')}] "
                f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')[:80]}..."
            )
        if len(result["segments"]) > 20:
            print(f"  ... and {len(result['segments']) - 20} more segments")

    if result.get("der_metrics"):
        der = result["der_metrics"]
        print("\n--- DER ---")
        print(f"DER: {der['der']:.6f} ({der['der_percent']:.2f}%)")
        for key in ("total", "correct", "confusion", "missed detection", "false alarm"):
            if key in der["details"]:
                print(f"{key}: {der['details'][key]:.3f}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference with LoRA Fine-tuned VibeVoice ASR"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="Path to base pretrained model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=None,
        help="Path to a single audio file to transcribe",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of audio files to transcribe in batch mode",
    )
    parser.add_argument(
        "--audio_glob",
        type=str,
        default=None,
        help="Optional glob used only on files directly under input_dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of files processed in batch mode",
    )
    parser.add_argument(
        "--context_info",
        type=str,
        default=None,
        help="Optional shared context info",
    )
    parser.add_argument(
        "--reference_json",
        type=str,
        default=None,
        help="Reference JSON label file for single-file DER evaluation",
    )
    parser.add_argument(
        "--reference_rttm",
        type=str,
        default=None,
        help="Reference RTTM file for single-file DER evaluation",
    )
    parser.add_argument(
        "--reference_json_dir",
        type=str,
        default=None,
        help="Directory of reference JSON files for batch DER evaluation",
    )
    parser.add_argument(
        "--reference_rttm_dir",
        type=str,
        default=None,
        help="Directory of reference RTTM files for batch DER evaluation",
    )
    parser.add_argument(
        "--der_collar",
        type=float,
        default=0.0,
        help="DER collar in seconds",
    )
    parser.add_argument(
        "--der_skip_overlap",
        action="store_true",
        help="Ignore overlap regions in DER evaluation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam size for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible audio-token sampling and generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="lora_inference.json",
        help="Single-file JSON output path, or batch summary JSON path",
    )
    parser.add_argument(
        "--output_rttm",
        type=str,
        default="",
        help="Optional single-file RTTM output path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_inference_outputs",
        help="Directory for batch per-file outputs",
    )
    ContentNoRepeatGenerationMixin.add_content_no_repeat_cli_args(parser)

    args = parser.parse_args()

    if bool(args.audio_file) == bool(args.input_dir):
        raise ValueError("Specify exactly one of --audio_file or --input_dir.")

    if args.seed is not None:
        set_global_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    batch_root = (
        Path(args.input_dir).expanduser().resolve()
        if args.input_dir is not None
        else None
    )
    items = _build_input_items(args)

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model, processor = load_lora_model(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    aggregate_metric = DiarizationErrorRate(
        collar=args.der_collar,
        skip_overlap=args.der_skip_overlap,
    )
    aggregate_der_files = 0
    results: List[Dict[str, Any]] = []

    output_dir = Path(args.output_dir).expanduser().resolve()
    batch_mode = args.input_dir is not None
    if batch_mode:
        (output_dir / "json").mkdir(parents=True, exist_ok=True)
        (output_dir / "rttm").mkdir(parents=True, exist_ok=True)

    for index, raw_item in enumerate(items, start=1):
        item = _prepare_item_metadata(raw_item, args, batch_root)
        audio_path = item["audio_path"]

        result = transcribe(
            model=model,
            processor=processor,
            audio_path=str(audio_path),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
            context_info=item["context_info"],
            device=args.device,
            seed=args.seed,
            content_no_repeat_ngram_size=args.content_no_repeat_ngram_size,
            content_no_repeat_decode_max_tokens=args.content_no_repeat_decode_max_tokens,
            content_no_repeat_debug=args.content_no_repeat_debug,
        )

        hypothesis_rttm_lines = _segments_to_rttm_lines(
            result.get("segments", []), item["file_id"]
        )
        reference_rttm_lines, reference_source = _build_reference_rttm_lines(item)

        der_metrics = None
        if reference_rttm_lines is not None:
            der_metrics = _calculate_der(
                reference_lines=reference_rttm_lines,
                hypothesis_lines=hypothesis_rttm_lines,
                uri=item["file_id"],
                collar=args.der_collar,
                skip_overlap=args.der_skip_overlap,
            )
            aggregate_metric(
                _annotation_from_rttm_lines(reference_rttm_lines, item["file_id"]),
                _annotation_from_rttm_lines(hypothesis_rttm_lines, item["file_id"]),
            )
            aggregate_der_files += 1

        payload = {
            "file_index": index,
            "audio_file": str(audio_path),
            "relative_path": item["relative_path"],
            "context_info": item["context_info"],
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "num_beams": args.num_beams,
            },
            "seed": args.seed,
            "raw_text": result["raw_text"],
            "segments": result["segments"],
            "prediction_rttm": "\n".join(hypothesis_rttm_lines),
            "reference_source": reference_source,
            "reference_rttm": (
                "\n".join(reference_rttm_lines)
                if reference_rttm_lines is not None
                else None
            ),
            "der_metrics": der_metrics,
        }
        results.append(payload)

        if batch_mode:
            relative = Path(item["relative_path"]).with_suffix("")
            _write_json(output_dir / "json" / relative.with_suffix(".json"), payload)
            prediction_rttm_text = "\n".join(hypothesis_rttm_lines)
            if prediction_rttm_text:
                prediction_rttm_text += "\n"
            _write_text(
                output_dir / "rttm" / relative.with_suffix(".rttm"),
                prediction_rttm_text,
            )

        status = (
            f"[{index}/{len(items)}] {audio_path.name}: "
            f"{len(result['segments'])} segments"
        )
        if der_metrics is not None:
            details = der_metrics.get("details", {})
            ref_speakers = len(der_metrics.get("ref_speakers", []))
            hyp_speakers = len(der_metrics.get("hyp_speakers", []))
            status += (
                f", DER={der_metrics['der_percent']:.2f}%"
                f", Speakers(ref/hyp)={ref_speakers}/{hyp_speakers}"
                f", Confusion={details.get('confusion', 0.0):.3f}"
                f", Miss={details.get('missed detection', 0.0):.3f}"
                f", False alarm={details.get('false alarm', 0.0):.3f}"
            )
        print(status)

    summary = {
        "mode": "batch" if batch_mode else "single",
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "device": args.device,
        "num_files": len(results),
        "der": _summarize_aggregate_der(
            aggregate_metric,
            aggregate_der_files,
            args.der_collar,
            args.der_skip_overlap,
        ),
        "results": results,
    }

    if batch_mode:
        summary_path = Path(args.output_json).expanduser().resolve()
        if summary_path.name == "lora_inference.json":
            summary_path = output_dir / "summary.json"
        _write_json(summary_path, summary)
        print(f"\nSaved batch summary to {summary_path}")
        print(f"Saved per-file JSON to {output_dir / 'json'}")
        print(f"Saved per-file RTTM to {output_dir / 'rttm'}")
        if summary["der"] is not None:
            print(
                f"Aggregate DER: {summary['der']['der']:.6f} "
                f"({summary['der']['der_percent']:.2f}%)"
            )
        return

    single_result = results[0]
    _print_single_result(single_result)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        _write_json(output_path, single_result)
        print(f"\nSaved transcription to {output_path}")

    rttm_path = _default_rttm_path(args.output_json, args.output_rttm)
    if rttm_path:
        rttm_text = single_result["prediction_rttm"]
        if rttm_text:
            rttm_text += "\n"
        _write_text(rttm_path.expanduser().resolve(), rttm_text)
        print(
            f"Saved RTTM ({len(single_result['segments'])} speaker segments) "
            f"to {rttm_path.expanduser().resolve()}"
        )


if __name__ == "__main__":
    main()
