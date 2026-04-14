#!/usr/bin/env python
"""
Batch Inference with LoRA Fine-tuned VibeVoice ASR Model

This script processes multiple audio files from a directory and outputs results
in the specified format.

Usage:
    python batch_inference_lora.py \
        --base_model microsoft/VibeVoice-ASR \
        --lora_path ./output/checkpoint-40 \
        --audio_dir /path/to/audio/directory \
        --output_dir ./results \
        --context_info "Hotwords: example"
"""

import argparse
import json
from pathlib import Path
import random
from typing import Optional, List, Dict, Any
import librosa

import numpy as np
import torch

from peft import PeftModel

from vibevoice.generation_mixin import ContentNoRepeatGenerationMixin
from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus', '.wma', '.aac'}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_lora_model(
    base_model_path: str,
    lora_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load base model and merge with LoRA weights."""
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


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}")
        return 0.0


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
    """Transcribe an audio file using the LoRA fine-tuned model."""
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
    except Exception as e:
        print(f"Warning: Failed to parse structured output: {e}")
        segments = []

    return {
        "raw_text": generated_text,
        "segments": segments,
    }


def format_output(
    audio_path: str,
    audio_duration: float,
    segments: List[Dict[str, Any]],
    context_info: Optional[str] = None,
) -> Dict[str, Any]:
    """Format output in the specified format."""
    formatted_segments = []
    for seg in segments:
        formatted_seg = {
            "speaker": seg.get("speaker_id", 0),
            "text": seg.get("text", ""),
            "start": seg.get("start_time", 0.0),
            "end": seg.get("end_time", 0.0),
        }
        formatted_segments.append(formatted_seg)

    customized_context = []
    if context_info:
        context_lines = context_info.strip().split('\n')
        for line in context_lines:
            if line.strip():
                customized_context.append(line.strip())

    output = {
        "audio_duration": audio_duration,
        "audio_path": Path(audio_path).name,
        "segments": formatted_segments,
    }

    if customized_context:
        output["customized_context"] = customized_context

    return output


def find_audio_files(directory: Path) -> List[Path]:
    """Find all audio files in the directory."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(directory.glob(f"*{ext}"))
        audio_files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(
        description="Batch Inference with LoRA Fine-tuned VibeVoice ASR"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="Path to base pretrained model",
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files to transcribe",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save transcription results",
    )
    parser.add_argument(
        "--context_info",
        type=str,
        default=None,
        help="Optional context info (e.g., 'Hotwords: Tea Brew, Aiden Host')",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=8192, help="Maximum tokens to generate"
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
        help="Beam size for generation (1 = greedy/sampling, >1 = beam search)",
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
        "--num_samples",
        type=int,
        default=None,
        help="Number of random audio files to process (default: process all files)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for file sampling (default: use current time)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already have output JSON files",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=None,
        help="Minimum audio duration in seconds (files shorter than this will be skipped)",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (files longer than this will be skipped)",
    )
    ContentNoRepeatGenerationMixin.add_content_no_repeat_cli_args(parser)

    args = parser.parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Filter by duration if requested
    if args.min_duration is not None or args.max_duration is not None:
        filtered_files = []
        for audio_file in audio_files:
            duration = get_audio_duration(str(audio_file))
            if duration == 0.0:
                continue
            if args.min_duration is not None and duration < args.min_duration:
                continue
            if args.max_duration is not None and duration > args.max_duration:
                continue
            filtered_files.append(audio_file)

        print(f"Filtered by duration: {len(filtered_files)} files remain")
        if args.min_duration is not None:
            print(f"  Min duration: {args.min_duration}s ({args.min_duration/60:.1f} min)")
        if args.max_duration is not None:
            print(f"  Max duration: {args.max_duration}s ({args.max_duration/60:.1f} min)")
        audio_files = filtered_files

    if not audio_files:
        print("No audio files match the criteria")
        return

    # Random sampling if requested
    if args.num_samples is not None and args.num_samples < len(audio_files):
        if args.random_seed is not None:
            random.seed(args.random_seed)
            print(f"Using random seed for sampling: {args.random_seed}")
        audio_files = random.sample(audio_files, args.num_samples)
        print(f"Randomly selected {args.num_samples} files")

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model, processor = load_lora_model(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    for idx, audio_file in enumerate(audio_files):
        print(f"\n{'='*60}")
        print(f"Processing {idx + 1}/{len(audio_files)}: {audio_file.name}")
        print(f"{'='*60}")

        # Check if output file already exists
        output_file = output_dir / f"{audio_file.stem}.json"
        if args.skip_existing and output_file.exists():
            print(f"Skipping (output already exists): {output_file}")
            continue

        try:
            audio_duration = get_audio_duration(str(audio_file))

            result = transcribe(
                model=model,
                processor=processor,
                audio_path=str(audio_file),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_beams=args.num_beams,
                context_info=args.context_info,
                device=args.device,
                seed=args.seed,
                content_no_repeat_ngram_size=args.content_no_repeat_ngram_size,
                content_no_repeat_decode_max_tokens=args.content_no_repeat_decode_max_tokens,
                content_no_repeat_debug=args.content_no_repeat_debug,
            )

            formatted_output = format_output(
                audio_path=str(audio_file),
                audio_duration=audio_duration,
                segments=result["segments"],
                context_info=args.context_info,
            )

            output_file.write_text(
                json.dumps(formatted_output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved result to {output_file}")

            if result["segments"]:
                print(f"Found {len(result['segments'])} segments")
                for seg in result["segments"][:3]:
                    print(
                        f"  [{seg.get('start_time', 'N/A'):.2f}s - {seg.get('end_time', 'N/A'):.2f}s] "
                        f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')[:60]}..."
                    )
                if len(result["segments"]) > 3:
                    print(f"  ... and {len(result['segments']) - 3} more segments")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
