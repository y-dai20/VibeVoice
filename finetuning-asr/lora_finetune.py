#!/usr/bin/env python
"""
VibeVoice ASR LoRA Fine-tuning Script

This script implements LoRA (Low-Rank Adaptation) fine-tuning for the VibeVoice ASR model.
It uses PEFT (Parameter-Efficient Fine-Tuning) library for efficient training.
"""

import json
import logging
import copy
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import asdict, dataclass, field, replace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import numpy as np

from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from vibevoice.generation_mixin import ContentNoRepeatGenerationMixin
from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.utils import parse_structured_generation
from pyannote.metrics.diarization import DiarizationErrorRate
from utils import annotation_from_rttm_string as rttm_to_annotation

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def uses_wandb(report_to: Any) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to.lower() == "wandb" or "wandb" in {
            item.strip().lower() for item in report_to.split(",") if item.strip()
        }
    if isinstance(report_to, (list, tuple, set)):
        return any(str(item).strip().lower() == "wandb" for item in report_to)
    return False


def _make_wandb_safe_key(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_value = "".join(safe_chars).strip("_.")
    return safe_value or "unknown"


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_path: str = field(
        default="microsoft/VibeVoice-ASR",
        metadata={
            "help": "Path to pretrained model (HuggingFace model ID or local path)"
        },
    )
    tokenizer_path: str = field(
        default="Qwen/Qwen2.5-1.5B",
        metadata={
            "help": "Path to tokenizer/language model used by VibeVoiceASRProcessor"
        },
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    data_dir: str = field(
        default="./toy_dataset", metadata={"help": "Directory containing training data"}
    )
    test_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing test data for periodic inference evaluation"
        },
    )
    validation_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing validation data for per-epoch/step evaluation"
        },
    )
    validation_split_ratio: float = field(
        default=0.0,
        metadata={
            "help": "If validation_data_dir is not set, reserve this fraction of data_dir for validation"
        },
    )
    max_audio_length: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum audio length in seconds (default: no limit)"},
    )
    use_customized_context: bool = field(
        default=True,
        metadata={
            "help": "Whether to use customized_context from JSON as additional context"
        },
    )
    skip_error_samples: bool = field(
        default=True,
        metadata={
            "help": "If True, dataset errors are logged and skipped instead of aborting training"
        },
    )
    content_no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": 'Apply no-repeat-ngram of this size only inside a JSON "Content" field during test inference'
        },
    )
    content_no_repeat_decode_max_tokens: int = field(
        default=2048,
        metadata={
            "help": 'How many recent tokens to decode when detecting the active JSON "Content" field during test inference'
        },
    )
    content_no_repeat_debug: bool = field(
        default=False,
        metadata={
            "help": 'Print debug logs when the JSON "Content" repetition guard is active during test inference'
        },
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""

    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha (scaling factor)"}
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_acoustic_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Also apply LoRA to supported layers inside the acoustic tokenizer"
        },
    )
    lora_semantic_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Also apply LoRA to supported layers inside the semantic tokenizer"
        },
    )


@dataclass
class OptunaArguments:
    """Arguments for Optuna hyperparameter search."""

    optuna_search_space: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to JSON search-space file. If set, run Optuna instead of a single training run."
            )
        },
    )
    optuna_n_trials: int = field(
        default=20, metadata={"help": "Number of Optuna trials"}
    )
    optuna_timeout: Optional[int] = field(
        default=None,
        metadata={"help": "Optional Optuna timeout in seconds"},
    )
    optuna_study_name: str = field(
        default="vibevoice_asr_lora",
        metadata={"help": "Optuna study name"},
    )
    optuna_storage: Optional[str] = field(
        default=None,
        metadata={"help": "Optuna storage URL. Defaults to sqlite under output_dir."},
    )
    optuna_metric: str = field(
        default="test_der",
        metadata={
            "help": "Objective metric: test_der or train_loss",
        },
    )
    optuna_direction: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optimization direction. Defaults to minimize for current metrics."
        },
    )
    optuna_output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base output directory for Optuna trials. Defaults to <output_dir>/optuna."
        },
    )
    optuna_sampler: str = field(
        default="tpe",
        metadata={"help": "Optuna sampler: tpe or random"},
    )
    optuna_seed: int = field(
        default=42, metadata={"help": "Seed for the Optuna sampler and per-trial seeds"}
    )


@dataclass
class VibeVoiceASRDataCollator:
    """
    Data collator for VibeVoice ASR fine-tuning.
    Handles batching of variable-length audio and text sequences.
    """

    processor: VibeVoiceASRProcessor
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features into model inputs.
        """
        # Separate inputs and labels
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        acoustic_mask_list = [f["acoustic_input_mask"] for f in features]
        speech_list = [f["speech"] for f in features]
        vae_tok_lens = [f["vae_tok_len"] for f in features]
        sample_idx_list = [f["sample_idx"] for f in features]

        # Determine max lengths
        max_seq_len = max(len(ids) for ids in input_ids_list)
        max_speech_len = max(len(s) for s in speech_list)
        max_vae_len = max(vae_tok_lens)

        batch_size = len(features)

        # Initialize padded tensors
        input_ids = torch.full(
            (batch_size, max_seq_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_seq_len), self.label_pad_token_id, dtype=torch.long
        )
        acoustic_input_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        speech_tensors = torch.zeros((batch_size, max_speech_len), dtype=torch.float32)
        speech_masks = torch.zeros((batch_size, max_vae_len), dtype=torch.bool)

        # Fill in the tensors (right padding for training)
        # Note: processor uses left padding for inference/generation, but training uses right padding
        for i, (ids, lbls, amask, speech, vae_len) in enumerate(
            zip(
                input_ids_list,
                labels_list,
                acoustic_mask_list,
                speech_list,
                vae_tok_lens,
            )
        ):
            seq_len = len(ids)

            # Right padding for input_ids and labels
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(lbls, dtype=torch.long)
            acoustic_input_mask[i, :seq_len] = torch.tensor(amask, dtype=torch.bool)

            # Speech tensors (right padding, zeros work as padding)
            speech_len = len(speech)
            speech_tensors[i, :speech_len] = torch.tensor(speech, dtype=torch.float32)
            speech_masks[i, :vae_len] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "acoustic_input_mask": acoustic_input_mask,
            "speech_tensors": speech_tensors,
            "speech_masks": speech_masks,
            "sample_idx": torch.tensor(sample_idx_list, dtype=torch.long),
        }


class VibeVoiceASRDataset(Dataset):
    """
    Dataset for VibeVoice ASR fine-tuning.

    Expected data format:
        - Audio files: .mp3, .wav, .flac, etc.
        - Label files: .json with matching name

    JSON format:
        {
            "audio_path": "0.mp3",
            "audio_duration": 351.73,
            "segments": [
                {
                    "speaker": 0,
                    "text": "Hey everyone, welcome back...",
                    "start": 0.0,
                    "end": 38.68
                },
                ...
            ],
            "customized_context": ["Tea Brew", "The property is near Meter Street."]  # optional
        }
    """

    def __init__(
        self,
        data_dir: str,
        processor: VibeVoiceASRProcessor,
        max_audio_length: Optional[float] = None,  # in seconds
        use_customized_context: bool = True,
        skip_error_samples: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing audio files and JSON labels
            processor: VibeVoice ASR processor
            max_audio_length: Maximum audio length in seconds (None = no limit)
            use_customized_context: Whether to include customized_context in prompt
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.use_customized_context = use_customized_context
        self.skip_error_samples = skip_error_samples

        self._invalid_indices: Set[int] = set()

        # Find all JSON files
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load and validate all samples from data directory."""
        samples = []

        for json_path in sorted(self.data_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Get audio path from JSON
                audio_filename = data.get("audio_path")
                if not audio_filename:
                    logger.warning(f"No audio_path specified in {json_path}")
                    continue

                audio_path = self.data_dir / audio_filename
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

                # Optional: filter by duration
                if self.max_audio_length is not None:
                    duration = data.get("audio_duration", float("inf"))
                    if duration > self.max_audio_length:
                        logger.info(
                            f"Skipping {json_path.stem}: duration {duration:.1f}s > max {self.max_audio_length}s"
                        )
                        continue

                samples.append(
                    {
                        "audio_path": str(audio_path),
                        "json_path": str(json_path),
                        "data": data,
                    }
                )

            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")
                continue

        return samples

    def _format_transcription(self, segments: List[Dict], audio_duration: float) -> str:
        """
        Format transcription segments into JSON output format.

        This matches the expected model output format used in training.
        """
        formatted_segments = []

        for seg in segments:
            formatted_seg = {}
            # Add timestamp
            formatted_seg["Start"] = round(seg["start"], 2)
            formatted_seg["End"] = round(seg["end"], 2)
            # Add speaker if available
            if "speaker" in seg:
                formatted_seg["Speaker"] = seg["speaker"]
            # Add content
            formatted_seg["Content"] = seg.get("text", "")
            formatted_segments.append(formatted_seg)

        # Return as compact JSON string (no spaces after separators)
        return json.dumps(formatted_segments, ensure_ascii=False, separators=(",", ":"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample for training, optionally skipping bad samples."""

        if not self.samples:
            raise RuntimeError("No samples available in dataset")

        if not self.skip_error_samples:
            return self._process_sample(idx, self.samples[idx])

        total = len(self.samples)
        attempts = 0
        current_idx = idx % total

        while attempts < total:
            if current_idx in self._invalid_indices:
                attempts += 1
                current_idx = (current_idx + 1) % total
                continue

            sample = self.samples[current_idx]
            try:
                return self._process_sample(current_idx, sample)
            except Exception as exc:  # pragma: no cover - defensive
                self._invalid_indices.add(current_idx)
                logger.warning(
                    "Skipping dataset sample %s due to error: %s",
                    sample.get("json_path", f"idx_{current_idx}"),
                    exc,
                )
                attempts += 1
                current_idx = (current_idx + 1) % total

        raise RuntimeError(
            "All dataset samples failed to process; cannot continue training."
        )

    def _process_sample(self, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single dataset sample and return model features."""

        data = sample["data"]
        audio_path = sample["audio_path"]

        logger.info(
            "Processing dataset sample %d/%d | audio=%s | json=%s | duration=%.2fs | segments=%d",
            idx + 1,
            len(self.samples),
            audio_path,
            sample["json_path"],
            data.get("audio_duration", float("nan")),
            len(data.get("segments", [])),
        )

        # Prepare context info (customized_context)
        context_info = None
        if self.use_customized_context and "customized_context" in data:
            customized_context = data["customized_context"]
            if customized_context:
                context_info = "\n".join(customized_context)

        # Process audio using the processor's internal method
        encoding = self.processor._process_single_audio(
            audio_path,
            sampling_rate=None,
            add_generation_prompt=True,
            use_streaming=True,
            context_info=context_info,
        )

        # Get the input tokens (system + user + generation prompt)
        input_ids = encoding["input_ids"]
        acoustic_input_mask = encoding["acoustic_input_mask"]
        speech = encoding["speech"]
        vae_tok_len = encoding["vae_tok_len"]

        # Format the target transcription
        target_text = self._format_transcription(
            data["segments"], data.get("audio_duration", len(speech) / 24000)
        )

        # Encode target using apply_chat_template to match training format
        # This adds the assistant role tokens (e.g., <|im_start|>assistant\n...<|im_end|>)
        target_tokens = self.processor.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": target_text}],
            tokenize=True,
            add_generation_prompt=False,
        )

        # Combine input and target
        full_input_ids = input_ids + target_tokens
        full_acoustic_mask = acoustic_input_mask + [0] * len(target_tokens)

        # Create labels: -100 for input tokens, actual tokens for target
        # We mask the input portion so loss is only computed on the response
        labels = [-100] * len(input_ids) + target_tokens

        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "acoustic_input_mask": full_acoustic_mask,
            "speech": speech,
            "vae_tok_len": vae_tok_len,
            "sample_idx": idx,
        }

    def mark_invalid(self, idx: int) -> None:
        self._invalid_indices.add(idx)


def _extract_json_filenames(dataset: Dataset) -> List[str]:
    """Extract JSON filenames represented by a dataset or subset."""
    if isinstance(dataset, VibeVoiceASRDataset):
        return [Path(sample["json_path"]).name for sample in dataset.samples]

    if isinstance(dataset, Subset) and isinstance(dataset.dataset, VibeVoiceASRDataset):
        base_dataset = dataset.dataset
        filenames = []
        for index in dataset.indices:
            sample = base_dataset.samples[index]
            filenames.append(Path(sample["json_path"]).name)
        return filenames

    return []


def extract_json_array(text: str) -> str:
    """
    Extract JSON array from text that may contain chat template artifacts.

    Args:
        text: Text that may contain JSON array with possible prefix/suffix

    Returns:
        Extracted JSON array string
    """
    text = text.strip()

    # Find the first '[' and last ']' to extract the JSON array
    start_idx = text.find("[")
    end_idx = text.rfind("]")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx : end_idx + 1]

    return text


def json_to_rttm(json_str: str, file_id: str = "audio") -> str:
    """
    Convert JSON transcription format to RTTM format.

    RTTM format: SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>

    Args:
        json_str: JSON string with segments [{"Start": 0.0, "End": 1.5, "Speaker": 0, "Content": "..."}]
        file_id: File identifier for RTTM

    Returns:
        RTTM formatted string
    """
    segments = parse_structured_generation(json_str)
    if not segments:
        logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
        return ""

    rttm_lines = []
    for seg in segments:
        if any(key not in seg for key in ("start_time", "end_time", "speaker_id")):
            continue

        start = seg["start_time"]
        end = seg["end_time"]
        duration = end - start
        speaker = seg["speaker_id"]

        rttm_line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{speaker} <NA> <NA>"
        rttm_lines.append(rttm_line)

    return "\n".join(rttm_lines)



def calculate_der(
    reference_rttm: str, hypothesis_rttm: str, uri: str = "audio"
) -> Optional[Dict[str, float]]:
    """
    Calculate Diarization Error Rate (DER) between reference and hypothesis.

    Args:
        reference_rttm: Ground truth RTTM string
        hypothesis_rttm: Predicted RTTM string
        uri: URI for the annotations

    Returns:
        Dictionary with DER metrics or None if calculation fails
    """
    try:
        reference = rttm_to_annotation(reference_rttm, uri)
        hypothesis = rttm_to_annotation(hypothesis_rttm, uri)

        if reference is None or hypothesis is None:
            return None

        metric = DiarizationErrorRate()
        der_value = metric(reference, hypothesis)

        components = metric.compute_components(reference, hypothesis)

        return {
            "DER": der_value,
            "confusion": components["confusion"],
            "false_alarm": components["false alarm"],
            "missed_detection": components["missed detection"],
            "total": components["total"],
        }
    except Exception as e:
        logger.warning(f"Error calculating DER: {e}")
        return None


def count_unique_speakers_in_rttm(rttm_str: str) -> int:
    speakers = set()

    for line in rttm_str.strip().split("\n"):
        if not line or not line.startswith("SPEAKER"):
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        speakers.add(parts[7])

    return len(speakers)


def _levenshtein_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def calculate_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    reference = reference or ""
    hypothesis = hypothesis or ""
    distance = _levenshtein_distance(reference, hypothesis)
    reference_length = max(len(reference), 1)
    return {
        "CER": distance / reference_length,
        "edit_distance": distance,
        "reference_length": len(reference),
        "hypothesis_length": len(hypothesis),
    }


def run_inference_on_test_set(
    model: nn.Module,
    processor: VibeVoiceASRProcessor,
    test_dataset: VibeVoiceASRDataset,
    max_samples: int = 5,
    device: str = "cuda",
    content_no_repeat_ngram_size: int = 0,
    content_no_repeat_decode_max_tokens: int = 2048,
    content_no_repeat_debug: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run inference on test dataset samples and return results.

    Args:
        model: The model to use for inference
        processor: The processor for audio processing
        test_dataset: Test dataset
        max_samples: Maximum number of samples to evaluate
        device: Device to run inference on
        seed: Random seed for reproducibility

    Returns:
        List of inference results with predictions and ground truth
    """
    model.eval()
    results = []
    logits_processor = (
        ContentNoRepeatGenerationMixin.build_content_no_repeat_logits_processor(
            tokenizer=processor.tokenizer,
            content_no_repeat_ngram_size=content_no_repeat_ngram_size,
            content_no_repeat_decode_max_tokens=content_no_repeat_decode_max_tokens,
            content_no_repeat_debug=content_no_repeat_debug,
        )
    )

    num_samples = min(max_samples, len(test_dataset))
    logger.info(f"Running inference on {num_samples} test samples...")

    with torch.no_grad():
        for idx in range(num_samples):
            try:
                sample = test_dataset.samples[idx]
                data = sample["data"]
                audio_path = sample["audio_path"]

                context_info = None
                if test_dataset.use_customized_context and "customized_context" in data:
                    customized_context = data["customized_context"]
                    if customized_context:
                        context_info = "\n".join(customized_context)

                encoding = processor._process_single_audio(
                    audio_path,
                    sampling_rate=None,
                    add_generation_prompt=True,
                    use_streaming=True,
                    context_info=context_info,
                )

                input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long).to(
                    device
                )
                acoustic_input_mask = torch.tensor(
                    [encoding["acoustic_input_mask"]], dtype=torch.bool
                ).to(device)
                speech_tensors = torch.tensor(
                    [encoding["speech"]], dtype=torch.float32
                ).to(device)
                speech_masks = torch.zeros(
                    (1, encoding["vae_tok_len"]), dtype=torch.bool
                ).to(device)
                speech_masks[0, : encoding["vae_tok_len"]] = True

                generation_kwargs = {
                    "input_ids": input_ids,
                    "acoustic_input_mask": acoustic_input_mask,
                    "speech_tensors": speech_tensors,
                    "speech_masks": speech_masks,
                    "max_new_tokens": 8192,
                    "temperature": 0.0,
                    "num_beams": 3,
                    "do_sample": False,
                    "pad_token_id": processor.pad_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                }
                if logits_processor is not None:
                    generation_kwargs["logits_processor"] = logits_processor

                outputs = model.generate(**generation_kwargs)

                predicted_text = processor.tokenizer.decode(
                    outputs[0][len(encoding["input_ids"]) :],
                    skip_special_tokens=True,
                )

                ground_truth = test_dataset._format_transcription(
                    data["segments"],
                    data.get("audio_duration", len(encoding["speech"]) / 24000),
                )

                # Convert to RTTM format
                file_id = Path(audio_path).stem
                pred_rttm = json_to_rttm(predicted_text, file_id)
                gt_rttm = json_to_rttm(ground_truth, file_id)
                predicted_speaker_count = count_unique_speakers_in_rttm(pred_rttm)
                ground_truth_speaker_count = count_unique_speakers_in_rttm(gt_rttm)

                # Calculate DER
                der_metrics = calculate_der(gt_rttm, pred_rttm, uri=file_id)
                cer_metrics = calculate_cer(ground_truth, predicted_text)

                result = {
                    "sample_idx": idx,
                    "audio_path": audio_path,
                    "prediction": predicted_text,
                    "ground_truth": ground_truth,
                    "prediction_rttm": pred_rttm,
                    "ground_truth_rttm": gt_rttm,
                    "predicted_speaker_count": predicted_speaker_count,
                    "ground_truth_speaker_count": ground_truth_speaker_count,
                    "der_metrics": der_metrics,
                    "cer_metrics": cer_metrics,
                }
                results.append(result)

                logger.info(f"Test sample {idx + 1}/{num_samples}:")
                logger.info(f"  Audio: {audio_path}")
                logger.info(f"  Prediction (full):\n{predicted_text[:100]}")
                logger.info(f"  Ground Truth (full):\n{ground_truth[:100]}")
                logger.info(
                    f"  Speaker count: predicted={predicted_speaker_count}, ground_truth={ground_truth_speaker_count}"
                )
                if der_metrics:
                    logger.info(
                        f"  DER: {der_metrics['DER']:.4f} (confusion: {der_metrics['confusion']:.4f}, "
                        f"false_alarm: {der_metrics['false_alarm']:.4f}, "
                        f"missed_detection: {der_metrics['missed_detection']:.4f})"
                    )
                logger.info(
                    f"  CER: {cer_metrics['CER']:.4f} "
                    f"(edit_distance: {cer_metrics['edit_distance']}, "
                    f"ref_len: {cer_metrics['reference_length']})"
                )

            except Exception as e:
                logger.warning(f"Error during inference on sample {idx}: {e}")
                continue

    model.train()
    return results


class TestInferenceCallback(TrainerCallback):
    """Callback to run inference on test set at specific steps and save weights."""

    def __init__(
        self,
        test_dataset: Optional[VibeVoiceASRDataset],
        processor: VibeVoiceASRProcessor,
        eval_steps: int = 100,
        max_test_samples: int = 5,
        save_weights: bool = True,
        content_no_repeat_ngram_size: int = 3,
        content_no_repeat_decode_max_tokens: int = 1024,
        content_no_repeat_debug: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the callback.

        Args:
            test_dataset: Test dataset for inference
            processor: Processor for inference
            eval_steps: Run inference every N steps
            max_test_samples: Maximum number of test samples to evaluate
            save_weights: Whether to save model weights at each evaluation
            seed: Random seed for reproducibility
        """
        self.test_dataset = test_dataset
        self.processor = processor
        self.eval_steps = eval_steps
        self.max_test_samples = max_test_samples
        self.save_weights = save_weights
        self.content_no_repeat_ngram_size = content_no_repeat_ngram_size
        self.content_no_repeat_decode_max_tokens = content_no_repeat_decode_max_tokens
        self.content_no_repeat_debug = content_no_repeat_debug
        self.seed = seed
        self.latest_der_summary: Optional[Dict[str, Any]] = None
        self.latest_cer_summary: Optional[Dict[str, Any]] = None
        self.best_average_der: Optional[float] = None
        self.best_der_step: Optional[int] = None
        self.best_average_cer: Optional[float] = None
        self.best_cer_step: Optional[int] = None

    def _run_and_persist(self, args, state, model) -> None:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running test inference at step {state.global_step}")
        logger.info(f"{'=' * 80}\n")

        device = next(model.parameters()).device
        results = run_inference_on_test_set(
            model=model,
            processor=self.processor,
            test_dataset=self.test_dataset,
            max_samples=self.max_test_samples,
            device=str(device),
            content_no_repeat_ngram_size=self.content_no_repeat_ngram_size,
            content_no_repeat_decode_max_tokens=self.content_no_repeat_decode_max_tokens,
            content_no_repeat_debug=self.content_no_repeat_debug,
            seed=self.seed,
        )

        results_dir = Path(args.output_dir) / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"step_{state.global_step}_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved test results to {results_file}")

        rttm_dir = results_dir / f"step_{state.global_step}_rttm"
        rttm_dir.mkdir(parents=True, exist_ok=True)

        text_dir = results_dir / f"step_{state.global_step}_text"
        text_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            file_id = Path(result["audio_path"]).stem

            pred_rttm_file = rttm_dir / f"{file_id}_pred.rttm"
            with open(pred_rttm_file, "w", encoding="utf-8") as f:
                f.write(result["prediction_rttm"])

            gt_rttm_file = rttm_dir / f"{file_id}_gt.rttm"
            with open(gt_rttm_file, "w", encoding="utf-8") as f:
                f.write(result["ground_truth_rttm"])

            pred_text_file = text_dir / f"{file_id}_prediction.txt"
            with open(pred_text_file, "w", encoding="utf-8") as f:
                f.write(f"Audio: {result['audio_path']}\n")
                f.write(f"Sample Index: {result['sample_idx']}\n")
                f.write(
                    f"Predicted Speaker Count: {result['predicted_speaker_count']}\n"
                )
                f.write(
                    f"Ground Truth Speaker Count: {result['ground_truth_speaker_count']}\n"
                )
                f.write("=" * 80 + "\n")
                f.write("PREDICTION:\n")
                f.write(result["prediction"] + "\n\n")
                f.write("=" * 80 + "\n")
                f.write("GROUND TRUTH:\n")
                f.write(result["ground_truth"] + "\n\n")
                if result.get("der_metrics"):
                    f.write("=" * 80 + "\n")
                    f.write("DER METRICS:\n")
                    f.write(f"  DER: {result['der_metrics']['DER']:.4f}\n")
                    f.write(f"  Confusion: {result['der_metrics']['confusion']:.4f}\n")
                    f.write(
                        f"  False Alarm: {result['der_metrics']['false_alarm']:.4f}\n"
                    )
                    f.write(
                        f"  Missed Detection: {result['der_metrics']['missed_detection']:.4f}\n"
                    )
                if result.get("cer_metrics"):
                    f.write("=" * 80 + "\n")
                    f.write("CER METRICS:\n")
                    f.write(f"  CER: {result['cer_metrics']['CER']:.4f}\n")
                    f.write(
                        f"  Edit Distance: {result['cer_metrics']['edit_distance']}\n"
                    )
                    f.write(
                        f"  Reference Length: {result['cer_metrics']['reference_length']}\n"
                    )

        logger.info(f"Saved RTTM files to {rttm_dir}")
        logger.info(f"Saved text results to {text_dir}")

        der_values = [r["der_metrics"] for r in results if r.get("der_metrics")]
        if der_values:
            avg_der = sum(m["DER"] for m in der_values) / len(der_values)
            avg_confusion = sum(m["confusion"] for m in der_values) / len(der_values)
            avg_false_alarm = sum(m["false_alarm"] for m in der_values) / len(
                der_values
            )
            avg_missed = sum(m["missed_detection"] for m in der_values) / len(
                der_values
            )

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Average DER Metrics (over {len(der_values)} samples):")
            logger.info(f"  DER: {avg_der:.4f}")
            logger.info(f"  Confusion: {avg_confusion:.4f}")
            logger.info(f"  False Alarm: {avg_false_alarm:.4f}")
            logger.info(f"  Missed Detection: {avg_missed:.4f}")
            logger.info(f"{'=' * 80}\n")

            per_file_der = []
            for result in results:
                der_metrics = result.get("der_metrics")
                if not der_metrics:
                    continue
                file_id = Path(result["audio_path"]).stem
                per_file_der.append(
                    {
                        "sample_idx": result["sample_idx"],
                        "audio_path": result["audio_path"],
                        "file_id": file_id,
                        "wandb_key": _make_wandb_safe_key(file_id),
                        "DER": der_metrics["DER"],
                        "confusion": der_metrics["confusion"],
                        "false_alarm": der_metrics["false_alarm"],
                        "missed_detection": der_metrics["missed_detection"],
                    }
                )

            der_summary = {
                "step": state.global_step,
                "num_samples": len(der_values),
                "average_DER": avg_der,
                "average_confusion": avg_confusion,
                "average_false_alarm": avg_false_alarm,
                "average_missed_detection": avg_missed,
                "per_file_DER": per_file_der,
            }
            self.latest_der_summary = der_summary
            if self.best_average_der is None or avg_der < self.best_average_der:
                self.best_average_der = avg_der
                self.best_der_step = int(state.global_step)
            der_summary["best_average_DER_so_far"] = self.best_average_der
            der_summary["best_average_DER_step"] = self.best_der_step

            der_summary_file = (
                results_dir / f"step_{state.global_step}_der_summary.json"
            )
            with open(der_summary_file, "w", encoding="utf-8") as f:
                json.dump(der_summary, f, ensure_ascii=False, indent=2)

        cer_values = [r["cer_metrics"] for r in results if r.get("cer_metrics")]
        if cer_values:
            avg_cer = sum(m["CER"] for m in cer_values) / len(cer_values)
            avg_edit_distance = sum(m["edit_distance"] for m in cer_values) / len(
                cer_values
            )
            avg_reference_length = sum(m["reference_length"] for m in cer_values) / len(
                cer_values
            )

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Average CER Metrics (over {len(cer_values)} samples):")
            logger.info(f"  CER: {avg_cer:.4f}")
            logger.info(f"  Edit Distance: {avg_edit_distance:.2f}")
            logger.info(f"  Reference Length: {avg_reference_length:.2f}")
            logger.info(f"{'=' * 80}\n")

            cer_summary = {
                "step": state.global_step,
                "num_samples": len(cer_values),
                "average_CER": avg_cer,
                "average_edit_distance": avg_edit_distance,
                "average_reference_length": avg_reference_length,
            }
            self.latest_cer_summary = cer_summary
            if self.best_average_cer is None or avg_cer < self.best_average_cer:
                self.best_average_cer = avg_cer
                self.best_cer_step = int(state.global_step)
            cer_summary["best_average_CER_so_far"] = self.best_average_cer
            cer_summary["best_average_CER_step"] = self.best_cer_step

            cer_summary_file = (
                results_dir / f"step_{state.global_step}_cer_summary.json"
            )
            with open(cer_summary_file, "w", encoding="utf-8") as f:
                json.dump(cer_summary, f, ensure_ascii=False, indent=2)

            if uses_wandb(args.report_to):
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb_payload = {
                            "test_cer": avg_cer,
                            "test_cer_avg_edit_distance": avg_edit_distance,
                            "test_cer_avg_reference_length": avg_reference_length,
                        }
                        if self.latest_der_summary is not None:
                            wandb_payload["test_der"] = self.latest_der_summary[
                                "average_DER"
                            ]
                            wandb_payload["test_der_confusion"] = (
                                self.latest_der_summary["average_confusion"]
                            )
                            wandb_payload["test_der_false_alarm"] = (
                                self.latest_der_summary["average_false_alarm"]
                            )
                            wandb_payload["test_der_missed_detection"] = (
                                self.latest_der_summary["average_missed_detection"]
                            )
                            for per_file_metrics in self.latest_der_summary.get(
                                "per_file_DER", []
                            ):
                                wandb_payload[
                                    f"test_der_by_file/{per_file_metrics['wandb_key']}"
                                ] = per_file_metrics["DER"]
                        wandb.log(wandb_payload, step=state.global_step)
                except Exception as e:
                    logger.warning("Failed to log test CER/DER to wandb: %s", e)

        if self.save_weights:
            checkpoint_dir = (
                Path(args.output_dir) / f"checkpoint-step-{state.global_step}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(checkpoint_dir)
            self.processor.save_pretrained(checkpoint_dir)

            logger.info(f"Saved model weights to {checkpoint_dir}")

        logger.info(f"\n{'=' * 80}\n")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if self.test_dataset is None:
            return

        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self._run_and_persist(args=args, state=state, model=model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.test_dataset is None or model is None:
            return
        if state.global_step <= 0:
            return
        if self.eval_steps <= 0 or state.global_step % self.eval_steps != 0:
            self._run_and_persist(args=args, state=state, model=model)


class SafeTrainer(Trainer):
    """Trainer that skips batches causing CUDA OOM instead of aborting training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skipped_oom_batches = 0

    @staticmethod
    def _is_cuda_oom(error: Exception) -> bool:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        message = str(error).lower()
        return "out of memory" in message and "cuda" in message

    def _handle_oom(
        self, sample_indices: Optional[torch.Tensor], error: Exception
    ) -> None:
        self._skipped_oom_batches += 1
        logger.warning(
            "Encountered CUDA OOM (batch #%d skipped): %s",
            self._skipped_oom_batches,
            error,
        )

        dataset = getattr(self, "train_dataset", None)
        if sample_indices is not None and isinstance(dataset, VibeVoiceASRDataset):
            for idx in sample_indices.detach().cpu().tolist():
                dataset.mark_invalid(int(idx))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optim = getattr(self, "optimizer", None)
        if optim is not None:
            zero_grad = getattr(optim, "zero_grad", None)
            if callable(zero_grad):
                zero_grad(set_to_none=True)

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        sample_indices = inputs.pop("sample_idx", None)
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not self._is_cuda_oom(error):
                raise
            self._handle_oom(sample_indices, error)
            return torch.tensor(0.0, device=self.args.device)


def setup_output_logging(output_dir: str, log_filename: str = "training.log") -> Path:
    """Mirror logs to a file in the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = output_path / log_filename
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
        ):
            return log_path

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging to %s", log_path)
    return log_path


def resolve_resume_from_checkpoint(
    resume_from_checkpoint: Optional[str], output_dir: str
) -> Optional[str]:
    if resume_from_checkpoint is None:
        return None

    checkpoint_value = resume_from_checkpoint.strip()
    if not checkpoint_value:
        return None

    if checkpoint_value.lower() in {"auto", "latest", "true"}:
        latest_checkpoint = get_last_checkpoint(output_dir)
        if latest_checkpoint is None:
            raise ValueError(
                f"No checkpoint found under output_dir={output_dir!r} to resume from."
            )
        return latest_checkpoint

    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    return str(checkpoint_path)


def save_argument_snapshot(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
    output_dir: str,
    dataset_summary: Optional[Dict[str, Any]] = None,
    log_path: Optional[str] = None,
) -> None:
    """Save only the 4 training argument groups as JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    args_payload = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "lora_args": asdict(lora_args),
        "training_args": training_args.to_dict(),
    }
    if dataset_summary is not None:
        args_payload["dataset_summary"] = dataset_summary
    if log_path is not None:
        args_payload["log_path"] = log_path

    args_json_path = output_path / "training_args.json"
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump(args_payload, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Saved argument snapshot to %s", args_json_path)


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    Create LoRA configuration for VibeVoice ASR model.

    We apply LoRA to the language model's attention layers and MLP,
    and also to the speech connectors that project acoustic/semantic
    features into the language-model hidden space, plus the output head.

    Args:
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Target Qwen2 attention and MLP layers, plus the speech connectors
        # and output head.
        # Connector module names are kept fully qualified so we do not
        # accidentally match unrelated fc1/fc2 layers elsewhere in the model.
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "acoustic_connector.fc1",
            # "acoustic_connector.fc2",
            # "semantic_connector.fc1",
            # "semantic_connector.fc2",
            # "lm_head",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # use_dora=True
    )


def resolve_lora_target_modules(
    model: nn.Module,
    base_target_modules: Optional[List[str]] = None,
    include_acoustic_tokenizer: bool = False,
    include_semantic_tokenizer: bool = False,
    lora_rank: Optional[int] = None,
) -> List[str]:
    """
    Build the final list of LoRA target modules for the loaded model.

    Tokenizer LoRA targets are collected from exact module names so we can
    safely include their nested Conv1d/Linear layers without broad suffix
    matches that could affect unrelated parts of the model.
    """
    target_modules = list(base_target_modules or [])
    tokenizer_prefixes: List[str] = []
    if include_acoustic_tokenizer:
        tokenizer_prefixes.append("model.acoustic_tokenizer.")
    if include_semantic_tokenizer:
        tokenizer_prefixes.append("model.semantic_tokenizer.")

    if not tokenizer_prefixes:
        return target_modules

    supported_module_types = (nn.Linear, nn.Conv1d)
    seen = set(target_modules)
    tokenizer_targets: List[str] = []
    skipped_grouped_convs: List[str] = []

    for module_name, module in model.named_modules():
        if not module_name.startswith(tuple(tokenizer_prefixes)):
            continue
        if not isinstance(module, supported_module_types):
            continue
        if isinstance(module, nn.Conv1d) and module.groups > 1:
            if lora_rank is None or (lora_rank % module.groups != 0):
                skipped_grouped_convs.append(
                    f"{module_name} (groups={module.groups}, rank={lora_rank})"
                )
                continue
        if module_name in seen:
            continue
        seen.add(module_name)
        tokenizer_targets.append(module_name)

    logger.info(
        "Resolved %d tokenizer LoRA target modules (%d total targets)",
        len(tokenizer_targets),
        len(seen),
    )
    if tokenizer_targets:
        logger.info("Tokenizer LoRA targets:\n%s", "\n".join(tokenizer_targets))
    if skipped_grouped_convs:
        logger.warning(
            "Skipped %d grouped Conv1d tokenizer modules that are incompatible with the current LoRA rank:\n%s",
            len(skipped_grouped_convs),
            "\n".join(skipped_grouped_convs),
        )

    target_modules.extend(tokenizer_targets)
    return target_modules


def setup_model_for_training(
    model_path: str,
    tokenizer_path: str,
    lora_config: LoraConfig,
    lora_args: Optional[LoraArguments] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> Tuple[nn.Module, VibeVoiceASRProcessor]:
    """
    Load and prepare model for LoRA training.

    Args:
        model_path: Path to pretrained model
        tokenizer_path: Path to tokenizer/language model
        lora_config: LoRA configuration
        device: Device to use
        dtype: Data type for model
        gradient_checkpointing: Whether to use gradient checkpointing

    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model from {model_path}")

    # Load processor
    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path, language_model_pretrained_name=tokenizer_path
    )

    # Load model
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if device != "auto":
        model = model.to(device)

    resolved_target_modules = resolve_lora_target_modules(
        model,
        base_target_modules=list(lora_config.target_modules or []),
        include_acoustic_tokenizer=(
            lora_args.lora_acoustic_tokenizer if lora_args is not None else False
        ),
        include_semantic_tokenizer=(
            lora_args.lora_semantic_tokenizer if lora_args is not None else False
        ),
        lora_rank=lora_config.r,
    )
    lora_config = get_lora_config(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=resolved_target_modules,
    )

    # Apply LoRA
    logger.info(
        f"Applying LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}"
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model, processor


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
    gradient_checkpointing: bool = True,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main training function for LoRA fine-tuning.

    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        lora_args: LoRA configuration arguments
        training_args: HuggingFace TrainingArguments
        gradient_checkpointing: Whether to use gradient checkpointing
    """
    log_path = setup_output_logging(training_args.output_dir)
    resolved_resume_from_checkpoint = resolve_resume_from_checkpoint(
        resume_from_checkpoint=resume_from_checkpoint,
        output_dir=training_args.output_dir,
    )
    if resolved_resume_from_checkpoint is not None:
        logger.info(
            "Resuming training from checkpoint %s", resolved_resume_from_checkpoint
        )
    if uses_wandb(training_args.report_to):
        if not training_args.run_name:
            training_args.run_name = Path(training_args.output_dir).resolve().name
        os.environ.setdefault("WANDB_NAME", training_args.run_name)
        os.environ.setdefault("WANDB_NOTES", "VibeVoice ASR LoRA fine-tuning")
        logger.info(
            "Weights & Biases logging enabled: run_name=%s project=%s",
            training_args.run_name,
            os.environ.get("WANDB_PROJECT", "(wandb default)"),
        )
    save_argument_snapshot(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args,
        output_dir=training_args.output_dir,
        log_path=str(log_path),
    )

    # Set seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # Setup LoRA config
    lora_config = get_lora_config(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model, processor = setup_model_for_training(
        model_path=model_args.model_path,
        tokenizer_path=model_args.tokenizer_path,
        lora_config=lora_config,
        lora_args=lora_args,
        device=device,
        dtype=dtype,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Create dataset
    full_train_dataset = VibeVoiceASRDataset(
        data_dir=data_args.data_dir,
        processor=processor,
        max_audio_length=data_args.max_audio_length,
        use_customized_context=data_args.use_customized_context,
        skip_error_samples=data_args.skip_error_samples,
    )

    train_dataset = full_train_dataset
    eval_dataset = None

    dataset_summary = {
        "train": {
            "data_dir": data_args.data_dir,
            "num_samples": len(full_train_dataset),
            "json_filenames": _extract_json_filenames(full_train_dataset),
        }
    }

    if data_args.validation_data_dir:
        logger.info("Loading validation dataset from %s", data_args.validation_data_dir)
        eval_dataset = VibeVoiceASRDataset(
            data_dir=data_args.validation_data_dir,
            processor=processor,
            max_audio_length=data_args.max_audio_length,
            use_customized_context=data_args.use_customized_context,
            skip_error_samples=data_args.skip_error_samples,
        )
        dataset_summary["validation"] = {
            "data_dir": data_args.validation_data_dir,
            "num_samples": len(eval_dataset),
            "json_filenames": _extract_json_filenames(eval_dataset),
        }
    elif data_args.validation_split_ratio > 0:
        if not 0.0 < data_args.validation_split_ratio < 1.0:
            raise ValueError("validation_split_ratio must be between 0 and 1.")
        dataset_size = len(full_train_dataset)
        if dataset_size < 2:
            raise ValueError(
                "At least 2 samples are required to create a validation split."
            )
        val_size = max(1, int(round(dataset_size * data_args.validation_split_ratio)))
        if val_size >= dataset_size:
            val_size = dataset_size - 1
        generator = torch.Generator().manual_seed(training_args.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_dataset = Subset(full_train_dataset, train_indices)
        eval_dataset = Subset(full_train_dataset, val_indices)
        dataset_summary["train"]["num_samples"] = len(train_dataset)
        dataset_summary["train"]["json_filenames"] = _extract_json_filenames(
            train_dataset
        )
        dataset_summary["validation"] = {
            "data_dir": data_args.data_dir,
            "split_ratio": data_args.validation_split_ratio,
            "num_samples": len(eval_dataset),
            "json_filenames": _extract_json_filenames(eval_dataset),
        }
        logger.info(
            "Split dataset into train=%d and validation=%d using seed=%d",
            len(train_dataset),
            len(eval_dataset),
            training_args.seed,
        )

    # Create test dataset if test_data_dir is provided
    test_dataset = None
    if data_args.test_data_dir:
        logger.info(f"Loading test dataset from {data_args.test_data_dir}")
        test_dataset = VibeVoiceASRDataset(
            data_dir=data_args.test_data_dir,
            processor=processor,
            max_audio_length=data_args.max_audio_length,
            use_customized_context=data_args.use_customized_context,
            skip_error_samples=data_args.skip_error_samples,
        )
        logger.info(f"Loaded {len(test_dataset)} test samples")
        dataset_summary["test"] = {
            "data_dir": data_args.test_data_dir,
            "num_samples": len(test_dataset),
            "json_filenames": _extract_json_filenames(test_dataset),
        }
    save_argument_snapshot(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args,
        output_dir=training_args.output_dir,
        dataset_summary=dataset_summary,
        log_path=str(log_path),
    )

    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return
    if eval_dataset is not None and len(eval_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    # Create data collator
    data_collator = VibeVoiceASRDataCollator(
        processor=processor,
        pad_token_id=processor.pad_id,
    )

    # Set some sensible defaults for audio training
    training_args.dataloader_num_workers = (
        0  # Audio loading can be tricky with multiprocessing
    )
    training_args.remove_unused_columns = False  # Keep all columns
    if eval_dataset is not None:
        training_args.do_eval = True
        if training_args.per_device_eval_batch_size == 8:
            training_args.per_device_eval_batch_size = (
                training_args.per_device_train_batch_size
            )
        if getattr(training_args, "evaluation_strategy", "no") == "no":
            training_args.evaluation_strategy = "steps"
        if (
            hasattr(training_args, "eval_strategy")
            and getattr(training_args, "eval_strategy", "no") == "no"
        ):
            training_args.eval_strategy = "steps"
        if training_args.eval_steps is None:
            training_args.eval_steps = (
                training_args.save_steps or training_args.logging_steps
            )
        if training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = "eval_loss"
        if training_args.greater_is_better is None:
            training_args.greater_is_better = False

    # Create callback for test inference
    callbacks = []
    test_callback: Optional[TestInferenceCallback] = None
    if test_dataset is not None:
        test_callback = TestInferenceCallback(
            test_dataset=test_dataset,
            processor=processor,
            eval_steps=training_args.save_steps,
            max_test_samples=10,
            save_weights=True,
            content_no_repeat_ngram_size=data_args.content_no_repeat_ngram_size,
            content_no_repeat_decode_max_tokens=data_args.content_no_repeat_decode_max_tokens,
            content_no_repeat_debug=data_args.content_no_repeat_debug,
            seed=training_args.seed,
        )
        callbacks.append(test_callback)
        logger.info(f"Test inference will run every {training_args.save_steps} steps")

    # Create trainer
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"  Num samples = {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"  Num validation samples = {len(eval_dataset)}")
    logger.info(f"  Num epochs = {training_args.num_train_epochs}")
    logger.info(f"  Batch size = {training_args.per_device_train_batch_size}")
    logger.info(
        f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    total_steps = (
        len(train_dataset)
        * int(training_args.num_train_epochs)
        // (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        )
    )
    logger.info(f"  Total optimization steps = {total_steps}")

    if resolved_resume_from_checkpoint is None:
        train_result = trainer.train()
    else:
        train_result = trainer.train(
            resume_from_checkpoint=resolved_resume_from_checkpoint
        )

    # Save final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    save_argument_snapshot(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args,
        output_dir=training_args.output_dir,
        dataset_summary=dataset_summary,
        log_path=str(log_path),
    )

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save processor config
    processor.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")
    result_payload: Dict[str, Any] = {
        "train_metrics": metrics,
        "dataset_summary": dataset_summary,
        "output_dir": training_args.output_dir,
    }
    if test_callback is not None:
        result_payload["test_der_latest"] = test_callback.latest_der_summary
        result_payload["test_der_best"] = (
            {
                "average_DER": test_callback.best_average_der,
                "step": test_callback.best_der_step,
            }
            if test_callback.best_average_der is not None
            else None
        )
        result_payload["test_cer_latest"] = test_callback.latest_cer_summary
        result_payload["test_cer_best"] = (
            {
                "average_CER": test_callback.best_average_cer,
                "step": test_callback.best_cer_step,
            }
            if test_callback.best_average_cer is not None
            else None
        )

    del trainer
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_payload


def load_optuna_search_space(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Optuna search space root must be a JSON object.")
    return payload


def build_optuna_sampler(name: str, seed: int):
    import optuna

    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    raise ValueError(f"Unsupported optuna sampler: {name}")


def suggest_optuna_value(trial, name: str, spec: Dict[str, Any]) -> Any:
    kind = str(spec.get("type", "")).lower()
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is None:
            return trial.suggest_float(name, low, high, log=log)
        return trial.suggest_float(name, low, high, step=float(step), log=log)
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind in {"categorical", "choice", "choices"}:
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{name}: choices must be a non-empty list.")
        return trial.suggest_categorical(name, choices)
    if kind == "bool":
        return trial.suggest_categorical(name, [False, True])
    raise ValueError(f"{name}: unsupported search space type {kind!r}")


def sample_optuna_overrides(
    trial, search_space: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    allowed_sections = {"model_args", "data_args", "lora_args", "training_args"}
    overrides: Dict[str, Dict[str, Any]] = {}
    for section, section_spec in search_space.items():
        if section not in allowed_sections:
            raise ValueError(f"Unsupported search-space section: {section}")
        if not isinstance(section_spec, dict):
            raise ValueError(f"Search-space section must be an object: {section}")
        section_overrides: Dict[str, Any] = {}
        for key, spec in section_spec.items():
            if not isinstance(spec, dict):
                raise ValueError(f"{section}.{key}: spec must be an object.")
            section_overrides[key] = suggest_optuna_value(
                trial, f"{section}.{key}", spec
            )
        if section_overrides:
            overrides[section] = section_overrides
    return overrides


def resolve_optuna_direction(metric: str, direction: Optional[str]) -> str:
    if direction is not None:
        if direction not in {"minimize", "maximize"}:
            raise ValueError("optuna_direction must be 'minimize' or 'maximize'.")
        return direction
    if metric in {"test_der", "train_loss"}:
        return "minimize"
    raise ValueError(f"Unsupported optuna metric: {metric}")


def apply_dataclass_overrides(instance, overrides: Dict[str, Any]):
    valid_fields = set(instance.__dataclass_fields__.keys())
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise ValueError(
            f"Unknown override fields for {type(instance).__name__}: {unknown}"
        )
    return replace(instance, **overrides)


def extract_optuna_metric(
    metric_name: str,
    train_result: Dict[str, Any],
) -> float:
    if metric_name == "train_loss":
        value = train_result.get("train_metrics", {}).get("train_loss")
        if value is None:
            raise RuntimeError("train_loss was not found in training metrics.")
        return float(value)
    if metric_name == "test_der":
        best_payload = train_result.get("test_der_best")
        if not best_payload or best_payload.get("average_DER") is None:
            raise RuntimeError(
                "test_der metric requires test_data_dir and at least one successful test inference pass."
            )
        return float(best_payload["average_DER"])
    raise ValueError(f"Unsupported optuna metric: {metric_name}")


def run_optuna(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
    optuna_args: OptunaArguments,
    gradient_checkpointing: bool = True,
) -> None:
    import optuna

    if optuna_args.optuna_search_space is None:
        raise ValueError("optuna_search_space is required to run Optuna.")

    metric_name = optuna_args.optuna_metric
    direction = resolve_optuna_direction(metric_name, optuna_args.optuna_direction)
    base_output_dir = Path(
        optuna_args.optuna_output_dir
        or (Path(training_args.output_dir).resolve() / "optuna")
    ).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    storage = optuna_args.optuna_storage
    if storage is None:
        storage = f"sqlite:///{(base_output_dir / 'optuna_study.db').resolve()}"

    search_space = load_optuna_search_space(optuna_args.optuna_search_space)
    study = optuna.create_study(
        study_name=optuna_args.optuna_study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=build_optuna_sampler(
            optuna_args.optuna_sampler, optuna_args.optuna_seed
        ),
    )

    logger.info(
        "Starting Optuna study name=%s metric=%s direction=%s trials=%d storage=%s",
        optuna_args.optuna_study_name,
        metric_name,
        direction,
        optuna_args.optuna_n_trials,
        storage,
    )

    def objective(trial) -> float:
        overrides = sample_optuna_overrides(trial, search_space)
        trial_output_dir = base_output_dir / f"trial_{trial.number:04d}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        local_model_args = apply_dataclass_overrides(
            copy.deepcopy(model_args), overrides.get("model_args", {})
        )
        local_data_args = apply_dataclass_overrides(
            copy.deepcopy(data_args), overrides.get("data_args", {})
        )
        local_lora_args = apply_dataclass_overrides(
            copy.deepcopy(lora_args), overrides.get("lora_args", {})
        )
        local_training_args = apply_dataclass_overrides(
            copy.deepcopy(training_args), overrides.get("training_args", {})
        )
        local_training_args = replace(
            local_training_args,
            output_dir=str(trial_output_dir),
            seed=int(optuna_args.optuna_seed + trial.number),
        )

        trial.set_user_attr("output_dir", str(trial_output_dir))
        trial.set_user_attr("overrides", overrides)

        result = train(
            model_args=local_model_args,
            data_args=local_data_args,
            lora_args=local_lora_args,
            training_args=local_training_args,
            gradient_checkpointing=gradient_checkpointing,
        )
        objective_value = extract_optuna_metric(metric_name, result)
        trial.set_user_attr("objective_value", objective_value)
        return objective_value

    study.optimize(
        objective,
        n_trials=optuna_args.optuna_n_trials,
        timeout=optuna_args.optuna_timeout,
    )

    summary = {
        "study_name": study.study_name,
        "metric": metric_name,
        "direction": direction,
        "storage": storage,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    summary_path = base_output_dir / "best_trial.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved Optuna summary to %s", summary_path)
    logger.info(
        "Best trial summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2)
    )


def main():
    # Use HfArgumentParser to parse all argument dataclasses
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            LoraArguments,
            TrainingArguments,
            OptunaArguments,
        )
    )
    model_args, data_args, lora_args, training_args, optuna_args = (
        parser.parse_args_into_dataclasses()
    )

    if optuna_args.optuna_search_space:
        if training_args.resume_from_checkpoint is not None:
            raise ValueError(
                "resume_from_checkpoint is not supported when running Optuna trials."
            )
        run_optuna(
            model_args=model_args,
            data_args=data_args,
            lora_args=lora_args,
            training_args=training_args,
            optuna_args=optuna_args,
        )
    else:
        train(
            model_args=model_args,
            data_args=data_args,
            lora_args=lora_args,
            training_args=training_args,
            resume_from_checkpoint=training_args.resume_from_checkpoint,
        )


if __name__ == "__main__":
    main()
