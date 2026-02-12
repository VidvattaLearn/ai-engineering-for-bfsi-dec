# Architecture

Two modes are supported:

1. Sandwich pipeline (STT to agent to TTS).
2. Real-time proxy (Azure GPT-4o Realtime WebSocket).

The frontend streams audio to the backend using WebSockets. The backend performs model calls and returns transcript and audio events.
