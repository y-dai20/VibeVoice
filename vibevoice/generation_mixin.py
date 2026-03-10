from __future__ import annotations

from datetime import datetime
from typing import Optional

import torch
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    _calc_banned_ngram_tokens,
)


class ContentNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Apply no-repeat-ngram only while generating a JSON Content field."""

    def __init__(
        self,
        tokenizer,
        ngram_size: int,
        decode_max_tokens: int = 2048,
        debug: bool = False,
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        if not isinstance(decode_max_tokens, int) or decode_max_tokens <= 0:
            raise ValueError(
                "`decode_max_tokens` has to be a strictly positive integer, "
                f"but is {decode_max_tokens}"
            )
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        self.decode_max_tokens = decode_max_tokens
        self.debug = debug
        self._debug_calls = 0
        self._debug_hits = 0

    @staticmethod
    def _extract_open_content_text(decoded_text: str) -> Optional[str]:
        marker = '"Content":"'
        marker_index = decoded_text.rfind(marker)
        if marker_index == -1:
            return None

        content_start = marker_index + len(marker)
        escaped = False
        for index in range(content_start, len(decoded_text)):
            char = decoded_text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                return None
        return decoded_text[content_start:]

    @staticmethod
    def _debug_timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self._debug_calls += 1
        num_batch_hypotheses = scores.shape[0]
        scores_processed = scores.clone()
        for i in range(num_batch_hypotheses):
            decode_ids = input_ids[i, -self.decode_max_tokens :]
            decoded_text = self.tokenizer.decode(
                decode_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            content_text = self._extract_open_content_text(decoded_text)
            if content_text is None:
                if self.debug:
                    print(
                        f"[{self._debug_timestamp()}][ContentNoRepeat] "
                        f"content_text is None. {decoded_text[-80:]}"
                    )
                continue

            if self.debug:
                preview = content_text[-80:].replace("\n", "\\n")
                print(
                    f"[{self._debug_timestamp()}][ContentNoRepeat] "
                    f"step={self._debug_calls} batch={i} "
                    f"content_chars={len(content_text)} tail={preview!r}"
                )

            content_ids = self.tokenizer.encode(
                content_text,
                add_special_tokens=False,
            )
            if len(content_ids) + 1 < self.ngram_size:
                continue

            content_input_ids = input_ids.new_tensor(content_ids).unsqueeze(0)
            banned_batch_tokens = _calc_banned_ngram_tokens(
                self.ngram_size,
                content_input_ids,
                1,
                content_input_ids.shape[-1],
            )
            banned_tokens = banned_batch_tokens[0]
            if banned_tokens:
                self._debug_hits += 1
                if self.debug:
                    print(
                        f"[{self._debug_timestamp()}][ContentNoRepeat] "
                        f"banned {len(banned_tokens)} token(s) "
                        f"for batch={i} at step={self._debug_calls}"
                    )
                scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class ContentNoRepeatGenerationMixin:
    @staticmethod
    def add_content_no_repeat_cli_args(parser) -> None:
        parser.add_argument(
            "--content_no_repeat_ngram_size",
            type=int,
            default=0,
            help='Apply no-repeat-ngram of this size only inside a JSON "Content" field (0 disables)',
        )
        parser.add_argument(
            "--content_no_repeat_decode_max_tokens",
            type=int,
            default=1024,
            help='How many recent tokens to decode when detecting the active JSON "Content" field',
        )
        parser.add_argument(
            "--content_no_repeat_debug",
            action="store_true",
            help='Print debug logs when the JSON "Content" repetition guard is active',
        )

    @staticmethod
    def build_content_no_repeat_logits_processor(
        tokenizer,
        content_no_repeat_ngram_size: int = 0,
        content_no_repeat_decode_max_tokens: int = 2048,
        content_no_repeat_debug: bool = False,
    ) -> Optional[LogitsProcessorList]:
        processors = LogitsProcessorList()
        if content_no_repeat_ngram_size and content_no_repeat_ngram_size > 0:
            processors.append(
                ContentNoRepeatNGramLogitsProcessor(
                    tokenizer=tokenizer,
                    ngram_size=content_no_repeat_ngram_size,
                    decode_max_tokens=content_no_repeat_decode_max_tokens,
                    debug=content_no_repeat_debug,
                )
            )
        return processors or None
