from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

import websockets


class AzureRealtimeClient:
    def __init__(self, endpoint: str, api_key: str, deployment: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment = deployment
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        base = self.endpoint
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        url = f"{base}/openai/realtime?api-version=2025-04-01-preview&deployment={self.deployment}"
        headers = {
            "api-key": self.api_key,
        }
        self.ws = await websockets.connect(url, extra_headers=headers)

        session_update = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"},
                "input_audio_transcription": {"model": "whisper-1", "language": "en"},
                "voice": "alloy",
                "instructions": "You are a helpful voice assistant. Respond in English only.",
            },
        }
        await self.ws.send(json.dumps(session_update))

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()

    async def send_audio(self, audio_base64: str) -> None:
        if not self.ws:
            return
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64,
        }
        await self.ws.send(json.dumps(message))

    async def commit_audio(self) -> None:
        if not self.ws:
            return
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        if not self.ws:
            return
        async for message in self.ws:
            yield json.loads(message)
