#!/usr/bin/env python
"""
Quantize VibeVoice ASR model with bitsandbytes and save it.

Usage:
    python quantize_vibevoice.py \
        --model_path microsoft/VibeVoice-ASR \
        --tokenizer_path Qwen/Qwen2.5-1.5B \
        --output_dir ./vibevoice-asr-4bit \
        --quantization 4bit
"""

import argparse
import importlib.metadata
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def _parse_dtype(dtype_name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return table[dtype_name]


def _ensure_bitsandbytes_installed() -> None:
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "bitsandbytes is not installed in the current Python environment.\n"
            "Install it in the same venv used to run this script.\n"
            "Example:\n"
            "  uv pip install bitsandbytes\n"
            "or:\n"
            "  pip install bitsandbytes"
        ) from exc
    print(f"Detected bitsandbytes=={version}")


def build_bnb_config(
    quantization: str, compute_dtype: torch.dtype
) -> BitsAndBytesConfig:
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(f"Unsupported quantization: {quantization}")


def quantize_and_save(
    model_path: str,
    tokenizer_path: str,
    output_dir: str,
    quantization: str,
    compute_dtype: torch.dtype,
    device_map: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor from: {model_path}")
    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path,
        language_model_pretrained_name=tokenizer_path,
    )

    print(f"Loading model in {quantization}...")
    bnb_config = build_bnb_config(
        quantization=quantization, compute_dtype=compute_dtype
    )

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Saving quantized model to: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and save VibeVoice ASR model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="Base model path or Hugging Face model id",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Tokenizer/language model path used by VibeVoiceASRProcessor",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save quantized model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default="4bit",
        help="Quantization mode",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Compute dtype used for 4-bit quantization",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for model loading (default: "auto")',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_bitsandbytes_installed()
    compute_dtype = _parse_dtype(args.compute_dtype)

    quantize_and_save(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        quantization=args.quantization,
        compute_dtype=compute_dtype,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
