from __future__ import annotations

import base64


def encode_audio_base64(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


def decode_audio_base64(data: str) -> bytes:
    return base64.b64decode(data)
