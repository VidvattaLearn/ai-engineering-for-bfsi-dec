from __future__ import annotations

from typing import AsyncIterator, Iterable, Optional, Union
import asyncio
import threading

try:
    from elevenlabs.client import ElevenLabs
except ImportError:  # pragma: no cover - fallback for older sdk layouts
    from elevenlabs import ElevenLabs


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self._client = ElevenLabs(api_key=api_key)

    def _build_stream(self, text: str) -> Iterable[bytes]:
        tts = self._client.text_to_speech
        if hasattr(tts, "stream"):
            return tts.stream(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2",
                output_format="pcm_16000",
            )
        if hasattr(tts, "convert_as_stream"):
            return tts.convert_as_stream(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2",
                output_format="pcm_16000",
            )
        audio = tts.convert(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_turbo_v2",
            output_format="pcm_16000",
        )
        return [audio] if audio else []

    async def _to_async(self, stream: Union[Iterable[bytes], AsyncIterator[bytes]]) -> AsyncIterator[bytes]:
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:  # type: ignore[func-returns-value]
                if chunk:
                    yield chunk
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[object]] = asyncio.Queue()

        def producer() -> None:
            try:
                for chunk in stream:  # type: ignore[assignment]
                    if chunk:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        header_buffer = bytearray()
        header_stripped = False
        raw_stream = self._build_stream(text)

        async for chunk in self._to_async(raw_stream):
            if not chunk:
                continue
            if header_stripped:
                yield chunk
                continue
            header_buffer.extend(chunk)
            if header_buffer.startswith(b"RIFF"):
                if len(header_buffer) < 44:
                    continue
                data = bytes(header_buffer[44:])
                header_buffer.clear()
                header_stripped = True
                if data:
                    yield data
            else:
                header_stripped = True
                yield bytes(header_buffer)
                header_buffer.clear()
