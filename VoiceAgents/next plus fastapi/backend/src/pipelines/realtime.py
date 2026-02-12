from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket

from ..services.azure_realtime import AzureRealtimeClient
from ..utils.audio import encode_audio_base64


class RealtimePipeline:
    def __init__(self, *, azure_endpoint: str, azure_key: str, azure_deployment: str) -> None:
        self._azure_endpoint = azure_endpoint
        self._azure_key = azure_key
        self._azure_deployment = azure_deployment

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client = AzureRealtimeClient(
            endpoint=self._azure_endpoint,
            api_key=self._azure_key,
            deployment=self._azure_deployment,
        )
        await client.connect()

        async def forward_events() -> None:
            async for event in client.events():
                event_type = event.get("type")
                if event_type == "input_audio_buffer.speech_started":
                    await websocket.send_json({"type": "speech_started"})
                elif event_type == "input_audio_buffer.speech_stopped":
                    await websocket.send_json({"type": "speech_ended"})
                elif event_type == "conversation.item.input_audio_transcription.delta":
                    await websocket.send_json({"type": "user_transcript_partial", "text": event.get("delta", "")})
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = (
                        event.get("transcript")
                        or event.get("text")
                        or event.get("item", {}).get("transcript")
                        or ""
                    )
                    await websocket.send_json({"type": "user_transcript_final", "text": text})
                elif event_type in ("response.text.delta", "response.output_text.delta"):
                    await websocket.send_json({"type": "assistant_text_delta", "text": event.get("delta", "")})
                elif event_type in ("response.text.done", "response.output_text.done"):
                    text = event.get("text", "")
                    if text:
                        await websocket.send_json({"type": "assistant_text_done", "text": text})
                elif event_type == "response.audio.delta":
                    await websocket.send_json({"type": "audio", "data": event.get("delta", "")})
                elif event_type == "response.done":
                    await websocket.send_json({"type": "done"})
                elif event_type == "error":
                    await websocket.send_json({"type": "error", "message": event.get("message", "Realtime error")})

        forward_task = asyncio.create_task(forward_events())

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    audio_b64 = encode_audio_base64(message["bytes"])
                    await client.send_audio(audio_b64)
                    continue
                if message.get("text"):
                    payload = json.loads(message["text"])
                    if payload.get("type") == "stop":
                        await client.commit_audio()
                        break
        finally:
            forward_task.cancel()
            await client.close()
