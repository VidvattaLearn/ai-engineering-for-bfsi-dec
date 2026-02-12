from __future__ import annotations

from dataclasses import dataclass
import asyncio
import io
import logging
import wave

try:
    from elevenlabs.client import ElevenLabs
except ImportError:  # pragma: no cover - fallback for older sdk layouts
    from elevenlabs import ElevenLabs


@dataclass
class STTEvent:
    type: str
    transcript: str
    confidence: float = 0.0


class ElevenLabsSTT:
    def __init__(self, api_key: str, sample_rate: int = 16000) -> None:
        self.api_key = api_key
        self.sample_rate = sample_rate
        self._audio_buffer = bytearray()
        self._client = ElevenLabs(api_key=api_key)

    async def connect(self) -> None:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is missing. Set it in backend/.env.")

    async def send_audio(self, audio_chunk: bytes) -> None:
        self._audio_buffer.extend(audio_chunk)

    async def close(self) -> None:
        return

    async def transcribe(self) -> STTEvent:
        logger = logging.getLogger("voiceagents.stt")
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(bytes(self._audio_buffer))
        wav_bytes = wav_buffer.getvalue()

        def run_transcription() -> object:
            return self._client.speech_to_text.convert(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model_id="scribe_v1",
                language_code="en",
            )

        logger.info("stt: sending request, bytes=%d", len(wav_bytes))
        result = await asyncio.to_thread(run_transcription)
        logger.info("stt: response received")

        text = ""
        confidence = 0.0
        if hasattr(result, "text"):
            text = getattr(result, "text", "") or ""
        elif isinstance(result, dict):
            text = result.get("text", "") or ""

        if hasattr(result, "confidence"):
            value = getattr(result, "confidence", 0.0)
            if isinstance(value, (float, int)):
                confidence = float(value)
        elif isinstance(result, dict):
            value = result.get("confidence", 0.0)
            if isinstance(value, (float, int)):
                confidence = float(value)

        return STTEvent(type="final", transcript=text, confidence=confidence)
