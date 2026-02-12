from __future__ import annotations

from typing import Iterable, List


SENTENCE_ENDINGS = (".", "!", "?", "\n")


def split_text_for_tts(text: str, min_chars: int = 80) -> List[str]:
    chunks: List[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if len(buffer) >= min_chars and buffer.endswith(SENTENCE_ENDINGS):
            chunks.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks
