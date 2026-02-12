from __future__ import annotations

import asyncio
import json
from typing import List, Dict

import logging

from fastapi import WebSocket

from ..agents.voice_agent import VoiceAgent
from ..services.elevenlabs_stt import ElevenLabsSTT
from ..services.elevenlabs_tts import ElevenLabsTTS
from ..utils.audio import encode_audio_base64
from ..utils.streaming import split_text_for_tts


class SandwichPipeline:
    def __init__(self, *, azure_endpoint: str, azure_key: str, azure_deployment: str, elevenlabs_key: str, elevenlabs_voice_id: str) -> None:
        self._azure_endpoint = azure_endpoint
        self._azure_key = azure_key
        self._azure_deployment = azure_deployment
        self._elevenlabs_key = elevenlabs_key
        self._elevenlabs_voice_id = elevenlabs_voice_id

    async def handle(self, websocket: WebSocket) -> None:
        logger = logging.getLogger("voiceagents.sandwich")
        agent = VoiceAgent(
            endpoint=self._azure_endpoint,
            api_key=self._azure_key,
            deployment=self._azure_deployment,
            system_prompt="You are a helpful voice assistant.",
        )
        stt = ElevenLabsSTT(self._elevenlabs_key)
        tts = ElevenLabsTTS(self._elevenlabs_key, self._elevenlabs_voice_id)

        await websocket.accept()
        await stt.connect()

        messages: List[Dict[str, str]] = []
        audio_buffer = bytearray()

        try:
            logger.info("sandwich: websocket accepted")
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.info("sandwich: websocket disconnect")
                    return
                if message.get("bytes"):
                    audio_buffer.extend(message["bytes"])
                    if len(audio_buffer) % (16000 * 2) == 0:
                        logger.info("sandwich: buffered %d bytes", len(audio_buffer))
                    continue
                if message.get("text"):
                    payload = json.loads(message["text"])
                    if payload.get("type") == "history":
                        history = payload.get("messages", [])
                        if isinstance(history, list):
                            messages = [m for m in history if isinstance(m, dict) and "role" in m and "content" in m]
                            logger.info("sandwich: received history, turns=%d", len(messages))
                        continue
                    if payload.get("type") == "end":
                        logger.info("sandwich: end received, total bytes=%d", len(audio_buffer))
                        break

            if not audio_buffer:
                logger.warning("sandwich: empty audio buffer")
                return

            await stt.send_audio(bytes(audio_buffer))
            logger.info("sandwich: audio sent to stt, bytes=%d", len(audio_buffer))
            event = await stt.transcribe()
            logger.info("sandwich: stt complete, chars=%d", len(event.transcript))
            await websocket.send_json({"type": "stt_final", "text": event.transcript})
            messages.append({"role": "user", "content": event.transcript})

            response_text = ""
            async for chunk in agent.stream_response(messages):
                response_text += chunk
                await websocket.send_json({"type": "agent_chunk", "text": chunk})

            messages.append({"role": "assistant", "content": response_text})
            logger.info("sandwich: agent complete, chars=%d", len(response_text))

            for tts_text in split_text_for_tts(response_text):
                logger.info("sandwich: tts chunk chars=%d", len(tts_text))
                async for audio_chunk in tts.stream(tts_text):
                    await websocket.send_json({
                        "type": "tts_audio",
                        "data": encode_audio_base64(audio_chunk),
                    })

            await websocket.send_json({"type": "complete"})
            logger.info("sandwich: complete sent")
        except Exception as exc:
            logger.exception("sandwich: error %s", exc)
            await websocket.send_json({"type": "error", "message": str(exc)})
