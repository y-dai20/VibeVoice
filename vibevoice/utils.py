"""Shared helpers for parsing structured ASR generation outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


StructuredParser = Callable[[str], list[dict[str, Any]]]


def normalize_generated_text(text: str) -> str:
    """Trim model-role prefixes from generated text."""
    normalized = (text or "").strip()
    normalized = re.sub(
        r"^\s*assistant\s*[:\n\r]*", "", normalized, flags=re.IGNORECASE
    )
    return normalized.strip()


def extract_json_payload(text: str) -> str | None:
    """Extract the first top-level JSON object or array from text."""
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
    for index in range(json_start, len(text)):
        if text[index] in "[{":
            bracket_count += 1
        elif text[index] in "]}":
            bracket_count -= 1
            if bracket_count == 0:
                json_end = index + 1
                break
    if json_end == -1:
        return None
    return text[json_start:json_end]


def _repair_common_json_issues(payload: str) -> str:
    """Repair common model-output breakages before JSON parsing."""
    repaired = payload.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r'("Speaker"\s*:\s*)"([^"]*),("Content"\s*:)',
        r'\1"\2",\3',
        repaired,
    )
    repaired = re.sub(
        r'("Speaker ID"\s*:\s*)"([^"]*),("Content"\s*:)',
        r'\1"\2",\3',
        repaired,
    )
    repaired = re.sub(
        r'(\{|,)\s*"?(?P<time>\d+(?:\.\d+)?)\s*,\s*"End(?: time)?"\s*:',
        r'\1 "Start":\g<time>,"End":',
        repaired,
    )
    return repaired


def _load_json_payload(payload: str) -> list[dict[str, Any]]:
    result = json.loads(payload)
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def canonicalize_segment_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map structured output keys to canonical segment fields."""
    key_mapping = {
        "Start time": "start_time",
        "Start": "start_time",
        "End time": "end_time",
        "End": "end_time",
        "Speaker ID": "speaker_id",
        "Speaker": "speaker_id",
        "Content": "text",
    }
    cleaned_result: list[dict[str, Any]] = []
    for item in items:
        cleaned_item: dict[str, Any] = {}
        for key, mapped_key in key_mapping.items():
            if key in item:
                cleaned_item[mapped_key] = item[key]
        if cleaned_item:
            cleaned_result.append(cleaned_item)
    return cleaned_result


def parse_structured_generation(
    text: str,
    structured_parser: StructuredParser | None = None,
) -> list[dict[str, Any]]:
    """Parse a model output string into canonical transcription segments."""
    normalized = normalize_generated_text(text)

    if structured_parser is not None:
        try:
            parsed = structured_parser(normalized)
            if parsed:
                return parsed
        except Exception:
            pass

    payload = extract_json_payload(normalized)
    if not payload:
        return []

    for candidate in (payload, _repair_common_json_issues(payload)):
        try:
            items = _load_json_payload(candidate)
            if items:
                return canonicalize_segment_items(items)
        except Exception:
            continue
    return []
