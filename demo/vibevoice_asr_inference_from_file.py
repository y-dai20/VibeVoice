#!/usr/bin/env python
"""
VibeVoice ASR Batch Inference Demo Script

This script supports batch inference for ASR model and compares results
between batch processing and single-sample processing.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import json
import re
from typing import List, Dict, Any, Optional

from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic kernels when available.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class VibeVoiceASRBatchInference:
    """Batch inference wrapper for VibeVoice ASR model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        seed: Optional[int] = None,
    ):
        """
        Initialize the ASR batch inference pipeline.

        Args:
            model_path: Path to the pretrained model
            device: Device to run inference on (cuda, mps, xpu, cpu, auto)
            dtype: Data type for model weights
            attn_implementation: Attention implementation to use ('flash_attention_2', 'sdpa', 'eager')
        """
        print(f"Loading VibeVoice ASR model from {model_path}")

        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path, language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        # Load model with specified attention implementation
        print(f"Using attention implementation: {attn_implementation}")
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.device = (
            device if device != "auto" else next(self.model.parameters()).device
        )
        self.dtype = dtype
        self.seed = seed
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    @staticmethod
    def _normalize_generated_text(text: str) -> str:
        normalized = (text or "").strip()
        normalized = re.sub(
            r"^\s*assistant\s*[:\n\r]*", "", normalized, flags=re.IGNORECASE
        )
        return normalized.strip()

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[str]:
        if not text:
            return None
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            if json_end > json_start:
                return text[json_start:json_end].strip()

        json_start = text.find("[")
        if json_start == -1:
            json_start = text.find("{")
        if json_start == -1:
            return None

        bracket_count = 0
        json_end = -1
        for i in range(json_start, len(text)):
            if text[i] in "[{":
                bracket_count += 1
            elif text[i] in "]}":
                bracket_count -= 1
                if bracket_count == 0:
                    json_end = i + 1
                    break
        if json_end == -1:
            return None
        return text[json_start:json_end]

    def _parse_segments_with_fallback(self, text: str) -> List[Dict[str, Any]]:
        try:
            parsed = self.processor.post_process_transcription(text)
            if parsed:
                return parsed
        except Exception as e:
            print(f"Warning: Failed to parse structured output: {e}")

        normalized = self._normalize_generated_text(text)
        payload = self._extract_json_payload(normalized)
        if not payload:
            return []

        try:
            result = json.loads(payload)
            if isinstance(result, dict):
                result = [result]
            if not isinstance(result, list):
                return []
        except Exception:
            return []

        key_mapping = {
            "Start time": "start_time",
            "Start": "start_time",
            "End time": "end_time",
            "End": "end_time",
            "Speaker ID": "speaker_id",
            "Speaker": "speaker_id",
            "Content": "text",
        }
        cleaned_result: List[Dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            cleaned_item: Dict[str, Any] = {}
            for key, mapped_key in key_mapping.items():
                if key in item:
                    cleaned_item[mapped_key] = item[key]
            if cleaned_item:
                cleaned_result.append(cleaned_item)
        return cleaned_result

    def _prepare_generation_config(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> dict:
        """Prepare generation configuration."""
        config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if repetition_penalty and repetition_penalty > 1.0:
            config["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            config["no_repeat_ngram_size"] = no_repeat_ngram_size

        # Beam search vs sampling
        if num_beams > 1:
            config["num_beams"] = num_beams
            config["do_sample"] = False  # Beam search doesn't use sampling
        else:
            config["do_sample"] = do_sample
            # Only set temperature and top_p when sampling is enabled
            if do_sample:
                config["temperature"] = temperature
                config["top_p"] = top_p

        return config

    def transcribe_batch(
        self,
        audio_inputs: List,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files/arrays in a single batch.

        Args:
            audio_inputs: List of audio file paths or (array, sampling_rate) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling

        Returns:
            List of transcription results
        """
        if len(audio_inputs) == 0:
            return []

        batch_size = len(audio_inputs)
        print(f"\nProcessing batch of {batch_size} audio(s)...")

        # Process all audio together
        inputs = self.processor(
            audio=audio_inputs,
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Print batch info
        print(f"  Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  Speech tensors shape: {inputs['speech_tensors'].shape}")
        print(f"  Attention mask shape: {inputs['attention_mask'].shape}")

        # Generate
        generation_config = self._prepare_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        start_time = time.time()

        if self.seed is not None:
            set_global_seed(self.seed)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)

        generation_time = time.time() - start_time

        # Decode outputs for each sample in the batch
        results = []
        input_length = inputs["input_ids"].shape[1]

        for i, audio_input in enumerate(audio_inputs):
            # Get generated tokens for this sample (excluding input tokens)
            generated_ids = output_ids[i, input_length:]

            # Remove padding tokens from the end
            # Find the first eos_token or pad_token
            eos_positions = (
                generated_ids == self.processor.tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[: eos_positions[0] + 1]

            generated_text = self.processor.decode(
                generated_ids, skip_special_tokens=True
            )
            normalized_text = self._normalize_generated_text(generated_text)
            transcription_segments = self._parse_segments_with_fallback(generated_text)

            # Get file name based on input type
            if isinstance(audio_input, str):
                file_name = audio_input
            elif isinstance(audio_input, dict) and "id" in audio_input:
                file_name = audio_input["id"]
            else:
                file_name = f"audio_{i}"

            results.append(
                {
                    "file": file_name,
                    "raw_text": generated_text,
                    "normalized_text": normalized_text,
                    "segments": transcription_segments,
                    "generation_time": generation_time / batch_size,
                }
            )

        print(f"  Total generation time: {generation_time:.2f}s")
        print(f"  Average time per sample: {generation_time / batch_size:.2f}s")

        return results

    def transcribe_with_batching(
        self,
        audio_inputs: List,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files/arrays with automatic batching.

        Args:
            audio_inputs: List of audio file paths or (array, sampling_rate) tuples
            batch_size: Number of samples per batch
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling

        Returns:
            List of transcription results
        """
        all_results = []

        # Process in batches
        for i in range(0, len(audio_inputs), batch_size):
            batch_inputs = audio_inputs[i : i + batch_size]
            print(f"\n{'=' * 60}")
            print(
                f"Processing batch {i // batch_size + 1}/{(len(audio_inputs) + batch_size - 1) // batch_size}"
            )

            batch_results = self.transcribe_batch(
                batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            all_results.extend(batch_results)

        return all_results


def print_result(result: Dict[str, Any]):
    """Pretty print a single transcription result."""
    print(f"\nFile: {result['file']}")
    print(f"Generation Time: {result['generation_time']:.2f}s")
    print("\n--- Raw Output ---")
    print(
        result["raw_text"][:500] + "..."
        if len(result["raw_text"]) > 500
        else result["raw_text"]
    )

    if result["segments"]:
        print(f"\n--- Structured Output ({len(result['segments'])} segments) ---")
        for seg in result["segments"][:50]:  # Show first 50 segments
            print(
                f"[{seg.get('start_time', 'N/A')} - {seg.get('end_time', 'N/A')}] "
                f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')}..."
            )
        if len(result["segments"]) > 50:
            print(f"  ... and {len(result['segments']) - 50} more segments")


def _json_default_serializer(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _parse_time_seconds(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
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


def _rttm_file_id(name: Any) -> str:
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

        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> speaker_{speaker_id} <NA> <NA>"
        )
    return lines


def _default_rttm_path(output_json: str, output_rttm: str) -> Optional[Path]:
    if output_rttm:
        return Path(output_rttm)
    if output_json:
        return Path(output_json).with_suffix(".rttm")
    return None


def load_dataset_and_concatenate(
    dataset_name: str,
    split: str,
    max_duration: float,
    num_audios: int,
    target_sr: int = 24000,
) -> Optional[List[np.ndarray]]:
    """
    Load a HuggingFace dataset and concatenate audio samples into long audio chunks.
    (Note, just for demo purpose, not for benchmark evaluation)

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'openslr/librispeech_asr')
        split: Dataset split to use (e.g., 'test', 'test.other')
        max_duration: Maximum duration in seconds for each concatenated audio
        num_audios: Number of concatenated audios to create
        target_sr: Target sample rate (default: 24000)

    Returns:
        List of concatenated audio arrays, or None if loading fails
    """
    try:
        from datasets import load_dataset
        import torchcodec  # just for decode audio in datasets
    except ImportError:
        print("Please install it with: pip install datasets torchcodec")
        return None

    print(f"\nLoading dataset: {dataset_name} (split: {split})")
    print(
        f"Will create {num_audios} concatenated audio(s), each up to {max_duration:.1f}s ({max_duration / 3600:.2f} hours)"
    )

    try:
        # Use streaming to avoid downloading the entire dataset
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        print("Dataset loaded in streaming mode")

        concatenated_audios = []  # List of concatenated audio metadata

        # Create multiple concatenated audios based on num_audios
        current_chunks = []
        current_duration = 0.0
        current_samples_used = 0
        sample_idx = 0

        for sample in dataset:
            if len(concatenated_audios) >= num_audios:
                break

            if "audio" not in sample:
                continue

            audio_data = sample["audio"]
            audio_array = audio_data["array"]
            sr = audio_data["sampling_rate"]

            # Resample if needed
            if sr != target_sr:
                duration = len(audio_array) / sr
                new_length = int(duration * target_sr)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array,
                )

            chunk_duration = len(audio_array) / target_sr

            # Check if adding this chunk exceeds max_duration
            if current_duration + chunk_duration > max_duration:
                remaining_duration = max_duration - current_duration
                if remaining_duration > 0.5:  # Only add if > 0.5s remaining
                    samples_to_take = int(remaining_duration * target_sr)
                    current_chunks.append(audio_array[:samples_to_take])
                    current_duration += remaining_duration
                    current_samples_used += 1

                # Save current concatenated audio and start a new one
                if current_chunks:
                    concatenated_audios.append(
                        {
                            "array": np.concatenate(current_chunks),
                            "duration": current_duration,
                            "samples_used": current_samples_used,
                        }
                    )
                    print(
                        f"  Created audio {len(concatenated_audios)}: {current_duration:.1f}s from {current_samples_used} samples"
                    )

                # Reset for next concatenated audio
                current_chunks = []
                current_duration = 0.0
                current_samples_used = 0

                if len(concatenated_audios) >= num_audios:
                    break

            current_chunks.append(audio_array)
            current_duration += chunk_duration
            current_samples_used += 1

            sample_idx += 1
            if sample_idx % 100 == 0:
                print(f"  Processed {sample_idx} samples...")

        # Don't forget the last batch if it has content
        if current_chunks and len(concatenated_audios) < num_audios:
            concatenated_audios.append(
                {
                    "array": np.concatenate(current_chunks),
                    "duration": current_duration,
                    "samples_used": current_samples_used,
                }
            )
            print(
                f"  Created audio {len(concatenated_audios)}: {current_duration:.1f}s from {current_samples_used} samples"
            )

        if not concatenated_audios:
            print("Warning: No audio samples found in dataset")
            return None

        # Extract arrays and print summary
        result = [a["array"] for a in concatenated_audios]
        total_duration = sum(a["duration"] for a in concatenated_audios)
        total_samples = sum(a["samples_used"] for a in concatenated_audios)
        print(
            f"\nCreated {len(result)} concatenated audio(s), total {total_duration:.1f}s ({total_duration / 60:.1f} min) from {total_samples} samples"
        )

        return result

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Batch Inference Demo")
    parser.add_argument(
        "--model_path", type=str, default="", help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--audio_files",
        type=str,
        nargs="+",
        required=False,
        help="Paths to audio files for transcription",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=False,
        help="Directory containing audio files for batch transcription",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="HuggingFace dataset name (e.g., 'openslr/librispeech_asr')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'test', 'test.other', 'test.clean')",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=3600.0,
        help="Maximum duration in seconds for concatenated dataset audio (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing multiple files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if torch.cuda.is_available()
        else (
            "xpu"
            if torch.backends.xpu.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        choices=["cuda", "cpu", "mps", "xpu", "auto"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy decoding)",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search. Use 1 for greedy/sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalty for repeated tokens (>1.0 suppresses repetition)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Disallow repeated n-grams of this size (0 disables)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation to use. 'auto' will select the best available for your device (flash_attention_2 for CUDA, sdpa for MPS/CPU/XPU)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output path to save all inference results as JSON",
    )
    parser.add_argument(
        "--output_rttm",
        type=str,
        default="",
        help="Optional output path to save speaker segments as RTTM (defaults to output_json with .rttm extension)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible audio-token sampling and generation",
    )

    args = parser.parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Auto-detect best attention implementation based on device
    if args.attn_implementation == "auto":
        if args.device == "cuda" and torch.cuda.is_available():
            try:
                import flash_attn

                args.attn_implementation = "flash_attention_2"
            except ImportError:
                print("flash_attn not installed, falling back to sdpa")
                args.attn_implementation = "sdpa"
        else:
            # MPS/XPU/CPU don't support flash_attention_2
            args.attn_implementation = "sdpa"
        print(f"Auto-detected attention implementation: {args.attn_implementation}")

    # Collect audio files
    audio_files = []
    concatenated_audio = None  # For storing concatenated dataset audio

    if args.audio_files:
        audio_files.extend(args.audio_files)

    if args.audio_dir:
        import glob

        for ext in ["*.wav", "*.mp3", "*.flac", "*.mp4", "*.m4a", "*.webm"]:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))

    if args.dataset:
        concatenated_audio = load_dataset_and_concatenate(
            dataset_name=args.dataset,
            split=args.split,
            max_duration=args.max_duration,
            num_audios=args.batch_size,
        )
        if concatenated_audio is None:
            return

    if len(audio_files) == 0 and concatenated_audio is None:
        print(
            "No audio files provided. Please specify --audio_files, --audio_dir, or --dataset."
        )
        return

    if audio_files:
        print(f"\nAudio files to process ({len(audio_files)}):")
        for f in audio_files:
            print(f"  - {f}")

    if concatenated_audio:
        print(f"\nConcatenated dataset audios: {len(concatenated_audio)} audio(s)")

    # Initialize model
    # Handle MPS device and dtype
    if args.device == "mps":
        model_dtype = torch.float32  # MPS works better with float32
    elif args.device == "xpu":
        model_dtype = torch.float32
    elif args.device == "cpu":
        model_dtype = torch.float32
    else:
        model_dtype = torch.bfloat16

    asr = VibeVoiceASRBatchInference(
        model_path=args.model_path,
        device=args.device,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
    )

    # If temperature is 0, use greedy decoding (no sampling)
    do_sample = args.temperature > 0

    # Combine all audio inputs
    all_audio_inputs = audio_files + (concatenated_audio or [])

    print("\n" + "=" * 80)
    print(f"Processing {len(all_audio_inputs)} audio(s)")
    print("=" * 80)

    all_results = asr.transcribe_with_batching(
        all_audio_inputs,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    for result in all_results:
        print("\n" + "-" * 60)
        print_result(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": args.model_path,
                    "device": args.device,
                    "attn_implementation": args.attn_implementation,
                    "generation_config": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_beams": args.num_beams,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                    },
                    "seed": args.seed,
                    "num_inputs": len(all_audio_inputs),
                    "results": all_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=_json_default_serializer,
            )
        print(f"\nSaved inference results to: {output_path}")

    rttm_path = _default_rttm_path(args.output_json, args.output_rttm)
    if rttm_path:
        rttm_lines: List[str] = []
        for result in all_results:
            rttm_lines.extend(
                _segments_to_rttm_lines(
                    result.get("segments", []), _rttm_file_id(result.get("file", "audio"))
                )
            )
        rttm_path.parent.mkdir(parents=True, exist_ok=True)
        rttm_text = "\n".join(rttm_lines)
        if rttm_text:
            rttm_text += "\n"
        rttm_path.write_text(rttm_text, encoding="utf-8")
        print(f"Saved RTTM ({len(rttm_lines)} speaker segments) to: {rttm_path}")


if __name__ == "__main__":
    main()
