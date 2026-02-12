# Voice Agents API

## Health

GET `/health`

Response:

```json
{"status": "ok"}
```

## WebSocket: Sandwich

Endpoint: `/ws/sandwich`

Client to server:

```json
{"type": "end"}
```

Binary frames: raw 16-bit PCM audio at 16kHz.

Server to client:

```json
{"type": "stt_partial", "text": "..."}
{"type": "stt_final", "text": "..."}
{"type": "agent_chunk", "text": "..."}
{"type": "tts_audio", "data": "<base64 pcm>"}
{"type": "complete"}
{"type": "error", "message": "..."}
```

## WebSocket: Real-time

Endpoint: `/ws/realtime`

Client to server:

```json
{"type": "stop"}
```

Binary frames: raw 16-bit PCM audio at 16kHz.

Server to client:

```json
{"type": "speech_started"}
{"type": "speech_ended"}
{"type": "transcript", "text": "..."}
{"type": "audio", "data": "<base64 pcm>"}
{"type": "done"}
{"type": "error", "message": "..."}
```
