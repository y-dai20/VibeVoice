#!/usr/bin/env python
"""
VibeVoice ASR LoRA Fine-tuning Script

This script implements LoRA (Low-Rank Adaptation) fine-tuning for the VibeVoice ASR model.
It uses PEFT (Parameter-Efficient Fine-Tuning) library for efficient training.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import asdict, dataclass, field
from io import StringIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        metadata={"help": "Directory containing test data for periodic inference evaluation"},
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


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""

    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha (scaling factor)"}
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


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
    start_idx = text.find('[')
    end_idx = text.rfind(']')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]

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
    # Extract JSON array from text that may contain chat template artifacts
    json_str = extract_json_array(json_str)

    try:
        segments = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
        return ""

    rttm_lines = []
    for seg in segments:
        start = seg.get("Start", 0.0)
        end = seg.get("End", 0.0)
        duration = end - start
        speaker = seg.get("Speaker", 0)

        rttm_line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{speaker} <NA> <NA>"
        rttm_lines.append(rttm_line)

    return "\n".join(rttm_lines)


def rttm_to_annotation(rttm_str: str, uri: str = "audio") -> Optional[Annotation]:
    """
    Convert RTTM string to pyannote Annotation object.

    Args:
        rttm_str: RTTM formatted string
        uri: URI for the annotation

    Returns:
        Annotation object or None if pyannote is not available
    """
    annotation = Annotation(uri=uri)

    for line in rttm_str.strip().split("\n"):
        if not line or not line.startswith("SPEAKER"):
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]

        segment = Segment(start, start + duration)
        annotation[segment] = speaker

    return annotation


def calculate_der(reference_rttm: str, hypothesis_rttm: str, uri: str = "audio") -> Optional[Dict[str, float]]:
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


def run_inference_on_test_set(
    model: nn.Module,
    processor: VibeVoiceASRProcessor,
    test_dataset: VibeVoiceASRDataset,
    max_samples: int = 5,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Run inference on test dataset samples and return results.

    Args:
        model: The model to use for inference
        processor: The processor for audio processing
        test_dataset: Test dataset
        max_samples: Maximum number of samples to evaluate
        device: Device to run inference on

    Returns:
        List of inference results with predictions and ground truth
    """
    model.eval()
    results = []

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

                input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long).to(device)
                acoustic_input_mask = torch.tensor(
                    [encoding["acoustic_input_mask"]], dtype=torch.bool
                ).to(device)
                speech_tensors = torch.tensor(
                    [encoding["speech"]], dtype=torch.float32
                ).to(device)
                speech_masks = torch.zeros((1, encoding["vae_tok_len"]), dtype=torch.bool).to(device)
                speech_masks[0, :encoding["vae_tok_len"]] = True

                outputs = model.generate(
                    input_ids=input_ids,
                    acoustic_input_mask=acoustic_input_mask,
                    speech_tensors=speech_tensors,
                    speech_masks=speech_masks,
                    max_new_tokens=8192,
                    temperature=0.0,
                    num_beams=3,
                    do_sample=False,
                    pad_token_id=processor.pad_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

                predicted_text = processor.tokenizer.decode(
                    outputs[0][len(encoding["input_ids"]):],
                    skip_special_tokens=True,
                )

                ground_truth = test_dataset._format_transcription(
                    data["segments"], data.get("audio_duration", len(encoding["speech"]) / 24000)
                )

                # Convert to RTTM format
                file_id = Path(audio_path).stem
                pred_rttm = json_to_rttm(predicted_text, file_id)
                gt_rttm = json_to_rttm(ground_truth, file_id)

                # Calculate DER
                der_metrics = calculate_der(gt_rttm, pred_rttm, uri=file_id)

                result = {
                    "sample_idx": idx,
                    "audio_path": audio_path,
                    "prediction": predicted_text,
                    "ground_truth": ground_truth,
                    "prediction_rttm": pred_rttm,
                    "ground_truth_rttm": gt_rttm,
                    "der_metrics": der_metrics,
                }
                results.append(result)

                logger.info(f"Test sample {idx + 1}/{num_samples}:")
                logger.info(f"  Audio: {audio_path}")
                logger.info(f"  Prediction (full):\n{predicted_text[:100]}")
                logger.info(f"  Ground Truth (full):\n{ground_truth[:100]}")
                if der_metrics:
                    logger.info(
                        f"  DER: {der_metrics['DER']:.4f} (confusion: {der_metrics['confusion']:.4f}, "
                        f"false_alarm: {der_metrics['false_alarm']:.4f}, "
                        f"missed_detection: {der_metrics['missed_detection']:.4f})"
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
    ):
        """
        Initialize the callback.

        Args:
            test_dataset: Test dataset for inference
            processor: Processor for inference
            eval_steps: Run inference every N steps
            max_test_samples: Maximum number of test samples to evaluate
            save_weights: Whether to save model weights at each evaluation
        """
        self.test_dataset = test_dataset
        self.processor = processor
        self.eval_steps = eval_steps
        self.max_test_samples = max_test_samples
        self.save_weights = save_weights

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if self.test_dataset is None:
            return

        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
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
            )

            results_dir = Path(args.output_dir) / "test_results"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON results
            results_file = results_dir / f"step_{state.global_step}_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved test results to {results_file}")

            # Save RTTM files and individual text results
            rttm_dir = results_dir / f"step_{state.global_step}_rttm"
            rttm_dir.mkdir(parents=True, exist_ok=True)

            text_dir = results_dir / f"step_{state.global_step}_text"
            text_dir.mkdir(parents=True, exist_ok=True)

            for result in results:
                file_id = Path(result["audio_path"]).stem

                # Save prediction RTTM
                pred_rttm_file = rttm_dir / f"{file_id}_pred.rttm"
                with open(pred_rttm_file, "w", encoding="utf-8") as f:
                    f.write(result["prediction_rttm"])

                # Save ground truth RTTM
                gt_rttm_file = rttm_dir / f"{file_id}_gt.rttm"
                with open(gt_rttm_file, "w", encoding="utf-8") as f:
                    f.write(result["ground_truth_rttm"])

                # Save full prediction text
                pred_text_file = text_dir / f"{file_id}_prediction.txt"
                with open(pred_text_file, "w", encoding="utf-8") as f:
                    f.write(f"Audio: {result['audio_path']}\n")
                    f.write(f"Sample Index: {result['sample_idx']}\n")
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
                        f.write(f"  False Alarm: {result['der_metrics']['false_alarm']:.4f}\n")
                        f.write(f"  Missed Detection: {result['der_metrics']['missed_detection']:.4f}\n")

            logger.info(f"Saved RTTM files to {rttm_dir}")
            logger.info(f"Saved text results to {text_dir}")

            # Calculate and log average DER metrics
            der_values = [r["der_metrics"] for r in results if r.get("der_metrics")]
            if der_values:
                avg_der = sum(m["DER"] for m in der_values) / len(der_values)
                avg_confusion = sum(m["confusion"] for m in der_values) / len(der_values)
                avg_false_alarm = sum(m["false_alarm"] for m in der_values) / len(der_values)
                avg_missed = sum(m["missed_detection"] for m in der_values) / len(der_values)

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Average DER Metrics (over {len(der_values)} samples):")
                logger.info(f"  DER: {avg_der:.4f}")
                logger.info(f"  Confusion: {avg_confusion:.4f}")
                logger.info(f"  False Alarm: {avg_false_alarm:.4f}")
                logger.info(f"  Missed Detection: {avg_missed:.4f}")
                logger.info(f"{'=' * 80}\n")

                # Save DER summary
                der_summary = {
                    "step": state.global_step,
                    "num_samples": len(der_values),
                    "average_DER": avg_der,
                    "average_confusion": avg_confusion,
                    "average_false_alarm": avg_false_alarm,
                    "average_missed_detection": avg_missed,
                }
                der_summary_file = results_dir / f"step_{state.global_step}_der_summary.json"
                with open(der_summary_file, "w", encoding="utf-8") as f:
                    json.dump(der_summary, f, ensure_ascii=False, indent=2)

            if self.save_weights:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-step-{state.global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(checkpoint_dir)
                self.processor.save_pretrained(checkpoint_dir)

                logger.info(f"Saved model weights to {checkpoint_dir}")

            logger.info(f"\n{'=' * 80}\n")


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


def save_argument_snapshot(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
    output_dir: str,
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
    following common practices for LLM fine-tuning.

    Args:
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Target Qwen2 attention and MLP layers
        # These are the common targets for language model fine-tuning
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def setup_model_for_training(
    model_path: str,
    tokenizer_path: str,
    lora_config: LoraConfig,
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

    # Freeze speech tokenizers (we only want to fine-tune the language model)
    for name, param in model.named_parameters():
        if "acoustic_tokenizer" in name or "semantic_tokenizer" in name:
            param.requires_grad = False
            logger.debug(f"Frozen: {name}")

    # Apply LoRA
    logger.info(
        f"Applying LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}"
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

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
):
    """
    Main training function for LoRA fine-tuning.

    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        lora_args: LoRA configuration arguments
        training_args: HuggingFace TrainingArguments
        gradient_checkpointing: Whether to use gradient checkpointing
    """
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
        device=device,
        dtype=dtype,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Create dataset
    train_dataset = VibeVoiceASRDataset(
        data_dir=data_args.data_dir,
        processor=processor,
        max_audio_length=data_args.max_audio_length,
        use_customized_context=data_args.use_customized_context,
        skip_error_samples=data_args.skip_error_samples,
    )

    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return

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

    # Create callback for test inference
    callbacks = []
    if test_dataset is not None:
        test_callback = TestInferenceCallback(
            test_dataset=test_dataset,
            processor=processor,
            eval_steps=training_args.save_steps,
            max_test_samples=5,
            save_weights=True,
        )
        callbacks.append(test_callback)
        logger.info(f"Test inference will run every {training_args.save_steps} steps")

    # Create trainer
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"  Num samples = {len(train_dataset)}")
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

    train_result = trainer.train()

    # Save final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    save_argument_snapshot(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args,
        output_dir=training_args.output_dir,
    )

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save processor config
    processor.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")

    return model, processor


def main():
    # Use HfArgumentParser to parse all argument dataclasses
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, TrainingArguments)
    )
    model_args, data_args, lora_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    # Run training
    train(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        training_args=training_args,
    )


if __name__ == "__main__":
    main()
