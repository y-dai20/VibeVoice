"""Shared RTTM and annotation utilities for VibeVoice ASR training and inference."""

from typing import Iterable

from pyannote.core import Annotation, Segment
from vibevoice.utils import (
    canonicalize_segment_items,
    extract_json_payload,
    normalize_generated_text,
    parse_structured_generation,
)


def annotation_from_rttm_lines(lines: Iterable[str], uri: str = "audio") -> Annotation:
    """Convert an iterable of RTTM lines to a pyannote Annotation object."""
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


def annotation_from_rttm_string(rttm_str: str, uri: str = "audio") -> Annotation:
    """Convert an RTTM-formatted string to a pyannote Annotation object."""
    return annotation_from_rttm_lines(rttm_str.splitlines(), uri=uri)
