#!/usr/bin/env python
"""
Inference with LoRA Fine-tuned VibeVoice ASR Model

This script loads a LoRA fine-tuned model and runs inference.

Usage:
    python inference_lora.py \
        --base_model microsoft/VibeVoice-ASR \
        --lora_path ./output \
        --audio_file ./toy_dataset/0.mp3
"""

import argparse
import json
from pathlib import Path
import re
from typing import Optional, List, Dict, Any

import torch

from peft import PeftModel

from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def _parse_time_seconds(value) -> Optional[float]:
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


def _segments_to_rttm_lines(
    segments: List[Dict[str, Any]], file_id: str
) -> List[str]:
    lines: List[str] = []
    for seg in segments or []:
        speaker_id = seg.get("speaker_id")
        if speaker_id is None:
            continue

        start = _parse_time_seconds(seg.get("start_time"))
        end = _parse_time_seconds(seg.get("end_time"))
        if start is None or end is None:
            continue

        dur = end - start
        if dur <= 0:
            continue

        speaker_label = f"speaker_{speaker_id}"
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker_label} <NA> <NA>"
        )
    return lines


def _default_rttm_path(output_json: Optional[str], output_rttm: Optional[str]) -> Optional[Path]:
    if output_rttm:
        return Path(output_rttm)
    if output_json:
        return Path(output_json).with_suffix(".rttm")
    return None


def load_lora_model(
    base_model_path: str,
    lora_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load base model and merge with LoRA weights.

    Args:
        base_model_path: Path to base pretrained model
        lora_path: Path to LoRA adapter weights
        device: Device to load model on
        dtype: Data type for model

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model from {base_model_path}")

    # Load processor
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model_path, language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )

    # Load base model
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if device != "auto":
        model = model.to(device)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    # Optionally merge LoRA weights into base model for faster inference
    # model = model.merge_and_unload()

    model.eval()
    print("Model loaded successfully")

    return model, processor


def transcribe(
    model,
    processor,
    audio_path: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    context_info: str = None,
    device: str = "cuda",
):
    """
    Transcribe an audio file using the LoRA fine-tuned model.

    Args:
        model: The LoRA fine-tuned model
        processor: The processor
        audio_path: Path to audio file
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        context_info: Optional context info (e.g., hotwords)
        device: Device

    Returns:
        Transcription result
    """
    print(f"\nTranscribing: {audio_path}")

    # Process audio
    inputs = processor(
        audio=audio_path,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    # Move to device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # Generation config
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_config["temperature"] = temperature
        gen_config["top_p"] = 0.9

    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_config)

    # Decode
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    cleaned_text = generated_text.strip()
    if cleaned_text.lower().startswith("assistant"):
        cleaned_text = cleaned_text.split("\n", 1)[-1].strip()

    # Parse structured output
    try:
        segments = processor.post_process_transcription(cleaned_text)
    except Exception as e:
        print(f"Warning: Failed to parse structured output: {e}")
        segments = []

    return {
        "raw_text": generated_text,
        "segments": segments,
    }


def main():
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
        "--lora_path", type=str, required=True, help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--audio_file", type=str, required=True, help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--context_info",
        type=str,
        default=None,
        help="Optional context info (e.g., 'Hotwords: Tea Brew, Aiden Host')",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=12000, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
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
        help="Optional path to save transcription result as JSON",
    )
    parser.add_argument(
        "--output_rttm",
        type=str,
        default="",
        help="Optional path to save speaker segments as RTTM (defaults to output_json with .rttm extension)",
    )

    args = parser.parse_args()

    # Load model
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model, processor = load_lora_model(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    # Transcribe
    result = transcribe(
        model=model,
        processor=processor,
        audio_path=args.audio_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        context_info=args.context_info,
        device=args.device,
    )

    # Print results
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

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "audio_file": args.audio_file,
            "context_info": args.context_info,
            "raw_text": result["raw_text"],
            "segments": result["segments"],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved transcription to {output_path.resolve()}")

    rttm_path = _default_rttm_path(args.output_json, args.output_rttm)
    if rttm_path:
        rttm_lines = _segments_to_rttm_lines(
            result.get("segments", []), _rttm_file_id(args.audio_file)
        )
        rttm_path.parent.mkdir(parents=True, exist_ok=True)
        rttm_text = "\n".join(rttm_lines)
        if rttm_text:
            rttm_text += "\n"
        rttm_path.write_text(rttm_text, encoding="utf-8")
        print(
            f"Saved RTTM ({len(rttm_lines)} speaker segments) to {rttm_path.resolve()}"
        )


if __name__ == "__main__":
    main()
