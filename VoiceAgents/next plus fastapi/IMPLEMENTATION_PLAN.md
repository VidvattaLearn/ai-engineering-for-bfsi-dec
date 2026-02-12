# Voice Agents Implementation Plan

## Project Overview

This project implements a voice-enabled AI agent system with two distinct architectures:

1. **Sandwich Architecture (STT -> Agent -> TTS)**: Uses ElevenLabs for speech-to-text and text-to-speech, with Azure OpenAI for the agent logic
2. **Real-time Voice-to-Voice**: Uses Azure OpenAI GPT-4o Realtime API for end-to-end voice interactions

Both architectures will be accessible through a Next.js frontend with WebSocket support for real-time streaming.

---

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Next.js Frontend                                   │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │    Sandwich Mode UI         │  │      Real-time Mode UI              │   │
│  │  - Audio Recording          │  │  - WebRTC/WebSocket Connection      │   │
│  │  - Transcript Display       │  │  - Real-time Audio Stream           │   │
│  │  - Audio Playback           │  │  - Conversation Display             │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                                      │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │   Sandwich Pipeline         │  │   Real-time Proxy                   │   │
│  │  ┌───────────────────────┐  │  │  ┌─────────────────────────────┐   │   │
│  │  │ ElevenLabs STT        │  │  │  │ Azure OpenAI GPT-4o         │   │   │
│  │  │ (Speech to Text)      │  │  │  │ Realtime API                │   │   │
│  │  └───────────┬───────────┘  │  │  │ - Direct WebSocket          │   │   │
│  │              ▼              │  │  │ - Server VAD                 │   │   │
│  │  ┌───────────────────────┐  │  │  │ - Tool Calling              │   │   │
│  │  │ Azure OpenAI Agent    │  │  │  └─────────────────────────────┘   │   │
│  │  │ (LangGraph ReAct)     │  │  │                                     │   │
│  │  └───────────┬───────────┘  │  └─────────────────────────────────────┘   │
│  │              ▼              │                                             │
│  │  ┌───────────────────────┐  │                                             │
│  │  │ ElevenLabs TTS        │  │                                             │
│  │  │ (Text to Speech)      │  │                                             │
│  │  └───────────────────────┘  │                                             │
│  └─────────────────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend (Python)
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | FastAPI | Async API server with WebSocket support |
| Agent Framework | LangChain v1 (create_agent) | ReAct agent with tool calling |
| LLM (Sandwich) | Azure OpenAI GPT-4o | Text-based agent reasoning |
| LLM (Real-time) | Azure OpenAI GPT-4o-realtime | Voice-to-voice processing |
| STT | ElevenLabs Speech-to-Text | Convert speech to text |
| TTS | ElevenLabs Text-to-Speech | Convert text to speech |
| Async Runtime | asyncio | Concurrent stream processing |

### Frontend (Next.js)
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Next.js 14 (App Router) | React framework with SSR |
| Styling | Tailwind CSS | Utility-first CSS |
| Audio | Web Audio API | Audio capture and playback |
| Communication | WebSocket | Real-time bidirectional communication |
| State | React Context/Zustand | Client state management |

---

## Folder Structure

```
VoiceAgents/
├── IMPLEMENTATION_PLAN.md          # This document
├── backend/
│   ├── requirements.txt            # Python dependencies
│   ├── .env.example                # Environment variables template
│   ├── main.py                     # FastAPI application entry
│   ├── config.py                   # Configuration management
│   └── src/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py          # Pydantic models for API
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base_agent.py       # Base agent class
│       │   └── voice_agent.py      # Voice-enabled agent
│       ├── services/
│       │   ├── __init__.py
│       │   ├── elevenlabs_stt.py   # ElevenLabs STT client
│       │   ├── elevenlabs_tts.py   # ElevenLabs TTS client
│       │   └── azure_realtime.py   # Azure GPT-4o Realtime client
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── sandwich.py         # STT -> Agent -> TTS pipeline
│       │   └── realtime.py         # Real-time voice pipeline
│       └── utils/
│           ├── __init__.py
│           ├── audio.py            # Audio processing utilities
│           └── streaming.py        # Stream merging utilities
│
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── .env.local.example
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # Home page with mode selection
│   │   ├── globals.css
│   │   ├── sandwich/
│   │   │   └── page.tsx            # Sandwich mode interface
│   │   └── realtime/
│   │       └── page.tsx            # Real-time mode interface
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   └── WaveformVisualizer.tsx
│   │   ├── AudioRecorder.tsx       # Audio recording component
│   │   ├── TranscriptDisplay.tsx   # Shows STT results
│   │   ├── AgentResponse.tsx       # Shows agent responses
│   │   ├── AudioPlayer.tsx         # TTS audio playback
│   │   ├── ConversationHistory.tsx # Full conversation view
│   │   └── ModeSelector.tsx        # Switch between modes
│   ├── hooks/
│   │   ├── useAudioRecorder.ts     # Audio recording hook
│   │   ├── useWebSocket.ts         # WebSocket connection hook
│   │   └── useAudioPlayer.ts       # Audio playback hook
│   ├── lib/
│   │   ├── websocket.ts            # WebSocket client
│   │   └── audio-utils.ts          # Audio processing utilities
│   └── types/
│       └── index.ts                # TypeScript type definitions
│
└── docs/
    ├── api.md                      # API documentation
    ├── setup.md                    # Setup instructions
    └── architecture.md             # Detailed architecture docs
```

---

## Part 1: Sandwich Architecture (STT -> Agent -> TTS)

### Overview

The sandwich architecture processes voice in three sequential stages with concurrent streaming at each stage:

```
Audio Input ─┐
             │
             ▼
     ┌───────────────┐
     │  ElevenLabs   │──► Partial Transcripts (streaming)
     │     STT       │
     └───────┬───────┘
             │ Final Transcript
             ▼
     ┌───────────────┐
     │  Azure OpenAI │──► Token-by-token response (streaming)
     │    Agent      │
     └───────┬───────┘
             │ Text chunks
             ▼
     ┌───────────────┐
     │  ElevenLabs   │──► Audio chunks (streaming)
     │     TTS       │
     └───────┬───────┘
             │
             ▼
       Audio Output
```

### 1.1 ElevenLabs Speech-to-Text Integration

**File: `backend/src/services/elevenlabs_stt.py`**

```python
import asyncio
import json
import websockets
from typing import AsyncIterator
from dataclasses import dataclass

@dataclass
class STTEvent:
    """Event from STT processing"""
    type: str  # 'partial' or 'final'
    transcript: str
    confidence: float = 0.0

class ElevenLabsSTT:
    """
    ElevenLabs Speech-to-Text client using WebSocket streaming.

    Implements producer-consumer pattern:
    - Producer: Sends audio chunks to ElevenLabs
    - Consumer: Receives transcript events
    """

    def __init__(self, api_key: str, sample_rate: int = 16000):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.ws = None
        self._audio_queue = asyncio.Queue()
        self._closed = False

    async def connect(self):
        """Establish WebSocket connection to ElevenLabs STT"""
        url = "wss://api.elevenlabs.io/v1/speech-to-text/stream"
        headers = {"xi-api-key": self.api_key}

        self.ws = await websockets.connect(url, extra_headers=headers)

        # Send initial configuration
        config = {
            "type": "config",
            "sample_rate": self.sample_rate,
            "encoding": "pcm_16",
            "language": "en"
        }
        await self.ws.send(json.dumps(config))

    async def send_audio(self, audio_chunk: bytes):
        """Queue audio chunk for sending"""
        if not self._closed:
            await self._audio_queue.put(audio_chunk)

    async def close(self):
        """Signal end of audio stream"""
        self._closed = True
        await self._audio_queue.put(None)  # Sentinel value

    async def _send_audio_loop(self):
        """Internal loop to send queued audio"""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                # Send end-of-stream signal
                await self.ws.send(json.dumps({"type": "end"}))
                break
            await self.ws.send(chunk)

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """Yield transcript events from ElevenLabs"""
        async for message in self.ws:
            data = json.loads(message)

            if data.get("type") == "transcript":
                yield STTEvent(
                    type="partial" if data.get("is_partial") else "final",
                    transcript=data.get("text", ""),
                    confidence=data.get("confidence", 0.0)
                )
            elif data.get("type") == "error":
                raise Exception(f"STT Error: {data.get('message')}")
            elif data.get("type") == "done":
                break

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[STTEvent]:
        """
        Full transcription pipeline with concurrent send/receive.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            STTEvent objects with partial and final transcripts
        """
        await self.connect()

        async def send_audio():
            try:
                async for chunk in audio_stream:
                    await self.send_audio(chunk)
            finally:
                await self.close()

        # Run sender and receiver concurrently
        send_task = asyncio.create_task(send_audio())

        try:
            async for event in self.receive_events():
                yield event
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            await self.ws.close()
```

### 1.2 Azure OpenAI Agent Integration

**File: `backend/src/agents/voice_agent.py`**

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import AsyncIterator
from dataclasses import dataclass
import os

@dataclass
class AgentEvent:
    """Event from agent processing"""
    type: str  # 'chunk' or 'complete'
    text: str
    tool_call: dict = None

class VoiceAgent:
    """
    LangGraph-based agent for voice interactions.

    Features:
    - Token-level streaming for fast TTS handoff
    - Memory persistence across conversation turns
    - Voice-optimized output (no markdown, emojis)
    """

    SYSTEM_PROMPT = """You are a helpful voice assistant. Your responses will be
spoken aloud, so follow these guidelines:

1. Keep responses concise and conversational
2. Do not use emojis, special characters, or markdown formatting
3. Avoid bullet points; use natural flowing sentences
4. Spell out numbers and abbreviations when appropriate
5. Use simple sentence structures that are easy to follow when heard

Be friendly, helpful, and natural in your responses."""

    def __init__(self, tools: list = None):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            streaming=True
        )

        self.memory = MemorySaver()
        self.tools = tools or []

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.SYSTEM_PROMPT,
            checkpointer=self.memory
        )

    async def process(
        self,
        transcript: str,
        thread_id: str
    ) -> AsyncIterator[AgentEvent]:
        """
        Process user transcript and stream response tokens.

        Args:
            transcript: User's speech converted to text
            thread_id: Conversation thread ID for memory

        Yields:
            AgentEvent with text chunks or completion signal
        """
        config = {"configurable": {"thread_id": thread_id}}

        async for event in self.agent.astream(
            {"messages": [HumanMessage(content=transcript)]},
            config,
            stream_mode="messages"
        ):
            message, metadata = event

            if hasattr(message, 'content') and message.content:
                yield AgentEvent(type="chunk", text=message.content)

            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    yield AgentEvent(
                        type="tool_call",
                        text="",
                        tool_call=tool_call
                    )

        yield AgentEvent(type="complete", text="")
```

### 1.3 ElevenLabs Text-to-Speech Integration

**File: `backend/src/services/elevenlabs_tts.py`**

```python
import asyncio
import json
import base64
import websockets
from typing import AsyncIterator
from dataclasses import dataclass
import uuid

@dataclass
class TTSEvent:
    """Event from TTS processing"""
    type: str  # 'audio_chunk' or 'complete'
    audio: bytes = None
    duration_ms: int = 0

class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech client using WebSocket streaming.

    Features:
    - Real-time text-to-audio conversion
    - Streaming output for immediate playback
    - Voice customization support
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        model_id: str = "eleven_turbo_v2_5"
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.ws = None
        self._text_queue = asyncio.Queue()
        self._closed = False

    async def connect(self):
        """Establish WebSocket connection to ElevenLabs TTS"""
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"

        params = f"?model_id={self.model_id}&output_format=pcm_16000"

        headers = {"xi-api-key": self.api_key}

        self.ws = await websockets.connect(url + params, extra_headers=headers)

        # Send initial configuration
        config = {
            "text": " ",  # Initial space to prime the model
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "generation_config": {
                "chunk_length_schedule": [120, 160, 250, 290]
            },
            "xi_api_key": self.api_key
        }
        await self.ws.send(json.dumps(config))

    async def send_text(self, text: str):
        """Queue text chunk for synthesis"""
        if not self._closed:
            await self._text_queue.put(text)

    async def close(self):
        """Signal end of text stream"""
        self._closed = True
        await self._text_queue.put(None)

    async def _send_text_loop(self):
        """Internal loop to send queued text"""
        while True:
            text = await self._text_queue.get()
            if text is None:
                # Send end-of-stream signal
                await self.ws.send(json.dumps({"text": ""}))
                break
            await self.ws.send(json.dumps({"text": text}))

    async def receive_audio(self) -> AsyncIterator[TTSEvent]:
        """Yield audio chunks from ElevenLabs"""
        async for message in self.ws:
            data = json.loads(message)

            if "audio" in data and data["audio"]:
                audio_bytes = base64.b64decode(data["audio"])
                yield TTSEvent(
                    type="audio_chunk",
                    audio=audio_bytes
                )

            if data.get("isFinal"):
                yield TTSEvent(type="complete")
                break

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[TTSEvent]:
        """
        Full synthesis pipeline with concurrent send/receive.

        Args:
            text_stream: Async iterator of text chunks

        Yields:
            TTSEvent objects with audio chunks
        """
        await self.connect()

        async def send_text():
            try:
                async for text in text_stream:
                    await self.send_text(text)
            finally:
                await self.close()

        send_task = asyncio.create_task(send_text())

        try:
            async for event in self.receive_audio():
                yield event
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            await self.ws.close()
```

### 1.4 Sandwich Pipeline Orchestration

**File: `backend/src/pipelines/sandwich.py`**

```python
import asyncio
from typing import AsyncIterator, Union
from dataclasses import dataclass
from uuid import uuid4

from ..services.elevenlabs_stt import ElevenLabsSTT, STTEvent
from ..services.elevenlabs_tts import ElevenLabsTTS, TTSEvent
from ..agents.voice_agent import VoiceAgent, AgentEvent

@dataclass
class PipelineEvent:
    """Unified event type for the sandwich pipeline"""
    stage: str  # 'stt', 'agent', 'tts'
    event_type: str
    data: Union[str, bytes, dict]
    timestamp: float = 0.0

class SandwichPipeline:
    """
    Orchestrates the STT -> Agent -> TTS pipeline.

    Implements streaming at each stage for minimal latency:
    - STT streams partial transcripts while receiving audio
    - Agent streams tokens as they're generated
    - TTS streams audio chunks as text arrives
    """

    def __init__(
        self,
        elevenlabs_api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        agent_tools: list = None
    ):
        self.stt = ElevenLabsSTT(api_key=elevenlabs_api_key)
        self.tts = ElevenLabsTTS(
            api_key=elevenlabs_api_key,
            voice_id=voice_id
        )
        self.agent = VoiceAgent(tools=agent_tools)

    async def process(
        self,
        audio_stream: AsyncIterator[bytes],
        thread_id: str = None
    ) -> AsyncIterator[PipelineEvent]:
        """
        Process audio through the complete sandwich pipeline.

        Args:
            audio_stream: Incoming audio chunks from user
            thread_id: Conversation thread ID (auto-generated if None)

        Yields:
            PipelineEvent objects from each stage
        """
        thread_id = thread_id or str(uuid4())

        # Stage 1: Speech-to-Text
        final_transcript = ""

        async for stt_event in self.stt.transcribe_stream(audio_stream):
            yield PipelineEvent(
                stage="stt",
                event_type=stt_event.type,
                data=stt_event.transcript
            )

            if stt_event.type == "final":
                final_transcript = stt_event.transcript

        if not final_transcript:
            return

        # Stage 2 & 3: Agent -> TTS (concurrent)
        # Create a queue to pass agent tokens to TTS
        text_queue = asyncio.Queue()

        async def agent_producer():
            """Generate agent response and queue for TTS"""
            full_response = ""
            async for agent_event in self.agent.process(
                final_transcript,
                thread_id
            ):
                if agent_event.type == "chunk":
                    full_response += agent_event.text
                    await text_queue.put(agent_event.text)
                    yield PipelineEvent(
                        stage="agent",
                        event_type="chunk",
                        data=agent_event.text
                    )
                elif agent_event.type == "tool_call":
                    yield PipelineEvent(
                        stage="agent",
                        event_type="tool_call",
                        data=agent_event.tool_call
                    )

            await text_queue.put(None)  # Signal end
            yield PipelineEvent(
                stage="agent",
                event_type="complete",
                data=full_response
            )

        async def text_generator():
            """Yield text chunks from queue for TTS"""
            while True:
                text = await text_queue.get()
                if text is None:
                    break
                yield text

        # Run agent and TTS concurrently
        agent_task = asyncio.create_task(self._collect_events(agent_producer()))

        async for tts_event in self.tts.synthesize_stream(text_generator()):
            yield PipelineEvent(
                stage="tts",
                event_type=tts_event.type,
                data=tts_event.audio if tts_event.audio else b""
            )

        # Yield collected agent events
        agent_events = await agent_task
        for event in agent_events:
            yield event

    async def _collect_events(
        self,
        event_gen: AsyncIterator[PipelineEvent]
    ) -> list:
        """Collect events from an async generator"""
        events = []
        async for event in event_gen:
            events.append(event)
        return events
```

---

## Part 2: Real-time Voice-to-Voice Architecture

### Overview

The real-time architecture uses Azure OpenAI's GPT-4o Realtime API for direct voice-to-voice processing:

```
┌─────────────────────────────────────────────────────┐
│                    User Browser                      │
│  ┌────────────────────────────────────────────┐     │
│  │         WebSocket Connection                │     │
│  │  - Sends: PCM audio chunks                  │     │
│  │  - Receives: Audio + Transcripts            │     │
│  └────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   Backend Proxy                      │
│  - Authenticates with Azure                          │
│  - Manages session lifecycle                         │
│  - Handles tool calls if needed                      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│          Azure OpenAI GPT-4o Realtime               │
│  ┌────────────────────────────────────────────┐     │
│  │  - Server-side VAD (Voice Activity Det.)   │     │
│  │  - Speech recognition (built-in)           │     │
│  │  - LLM processing                          │     │
│  │  - Speech synthesis (built-in)             │     │
│  │  - Tool calling support                    │     │
│  └────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

### 2.1 Azure OpenAI Realtime Client

**File: `backend/src/services/azure_realtime.py`**

```python
import asyncio
import json
import base64
import websockets
from typing import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import os

class RealtimeEventType(Enum):
    """Types of events from the Realtime API"""
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_DONE = "response.done"
    ERROR = "error"

@dataclass
class RealtimeEvent:
    """Event from Azure OpenAI Realtime API"""
    type: RealtimeEventType
    data: dict = field(default_factory=dict)
    audio: bytes = None
    transcript: str = ""

class AzureRealtimeClient:
    """
    Azure OpenAI GPT-4o Realtime API client.

    Features:
    - WebSocket-based real-time communication
    - Server-side Voice Activity Detection (VAD)
    - Integrated STT, LLM, and TTS
    - Tool calling support
    """

    def __init__(
        self,
        deployment: str = None,
        endpoint: str = None,
        api_key: str = None,
        voice: str = "alloy",
        instructions: str = None
    ):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.voice = voice
        self.instructions = instructions or self._default_instructions()

        self.ws = None
        self._tools = []
        self._tool_handlers = {}

    def _default_instructions(self) -> str:
        return """You are a helpful voice assistant. Respond naturally and
conversationally. Keep responses concise since they will be spoken aloud.
Be friendly and helpful."""

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable
    ):
        """Register a tool for the realtime agent"""
        self._tools.append({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters
        })
        self._tool_handlers[name] = handler

    async def connect(self) -> None:
        """Establish WebSocket connection to Azure OpenAI Realtime API"""
        # Construct WebSocket URL
        base_url = self.endpoint.replace("https://", "wss://")
        url = (
            f"{base_url}/openai/realtime"
            f"?api-version=2025-04-01-preview"
            f"&deployment={self.deployment}"
        )

        headers = {"api-key": self.api_key}

        self.ws = await websockets.connect(url, extra_headers=headers)

        # Configure session
        session_config = {
            "type": "session.update",
            "session": {
                "voice": self.voice,
                "instructions": self.instructions,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": True
                },
                "tools": self._tools if self._tools else []
            }
        }

        await self.ws.send(json.dumps(session_config))

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio chunk to the API"""
        if self.ws:
            message = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_chunk).decode()
            }
            await self.ws.send(json.dumps(message))

    async def commit_audio(self) -> None:
        """Commit the audio buffer for processing"""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))

    async def cancel_response(self) -> None:
        """Cancel the current response (for interruption handling)"""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "response.cancel"
            }))

    async def receive_events(self) -> AsyncIterator[RealtimeEvent]:
        """Receive and yield events from the API"""
        async for message in self.ws:
            data = json.loads(message)
            event_type = data.get("type", "")

            try:
                evt_type = RealtimeEventType(event_type)
            except ValueError:
                continue  # Skip unknown event types

            event = RealtimeEvent(type=evt_type, data=data)

            # Extract audio if present
            if evt_type == RealtimeEventType.RESPONSE_AUDIO_DELTA:
                if "delta" in data:
                    event.audio = base64.b64decode(data["delta"])

            # Extract transcript if present
            if evt_type == RealtimeEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
                event.transcript = data.get("delta", "")

            # Handle tool calls
            if evt_type == RealtimeEventType.CONVERSATION_ITEM_CREATED:
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    await self._handle_tool_call(item)

            yield event

            if evt_type == RealtimeEventType.ERROR:
                break

    async def _handle_tool_call(self, item: dict) -> None:
        """Execute tool call and send result back"""
        name = item.get("name")
        call_id = item.get("call_id")
        arguments = json.loads(item.get("arguments", "{}"))

        if name in self._tool_handlers:
            try:
                result = await self._tool_handlers[name](**arguments)
                result_str = json.dumps(result) if isinstance(result, dict) else str(result)
            except Exception as e:
                result_str = f"Error: {str(e)}"

            # Send tool result
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result_str
                }
            }))

            # Trigger response generation
            await self.ws.send(json.dumps({
                "type": "response.create"
            }))

    async def close(self) -> None:
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.ws = None
```

### 2.2 Real-time Pipeline

**File: `backend/src/pipelines/realtime.py`**

```python
import asyncio
from typing import AsyncIterator
from dataclasses import dataclass

from ..services.azure_realtime import AzureRealtimeClient, RealtimeEvent, RealtimeEventType

@dataclass
class RealtimePipelineEvent:
    """Event from the realtime pipeline"""
    type: str  # 'speech_started', 'speech_ended', 'transcript', 'audio', 'done', 'error'
    data: any = None

class RealtimePipeline:
    """
    Manages the Azure OpenAI Realtime voice pipeline.

    Features:
    - Automatic VAD (Voice Activity Detection)
    - Concurrent audio send/receive
    - Interruption handling
    - Tool integration
    """

    def __init__(
        self,
        voice: str = "alloy",
        instructions: str = None,
        tools: list = None
    ):
        self.client = AzureRealtimeClient(
            voice=voice,
            instructions=instructions
        )
        self.tools = tools or []
        self._is_responding = False

    def register_tools(self):
        """Register any configured tools with the client"""
        for tool in self.tools:
            self.client.register_tool(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
                handler=tool["handler"]
            )

    async def start(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[RealtimePipelineEvent]:
        """
        Start the realtime voice pipeline.

        Args:
            audio_stream: Incoming audio chunks from user

        Yields:
            RealtimePipelineEvent objects
        """
        self.register_tools()
        await self.client.connect()

        # Start audio sender task
        send_task = asyncio.create_task(self._send_audio(audio_stream))

        try:
            async for event in self.client.receive_events():
                pipeline_event = self._convert_event(event)
                if pipeline_event:
                    yield pipeline_event

                # Track response state for interruption handling
                if event.type == RealtimeEventType.RESPONSE_AUDIO_DELTA:
                    self._is_responding = True
                elif event.type == RealtimeEventType.RESPONSE_DONE:
                    self._is_responding = False

                # Handle user interruption
                if event.type == RealtimeEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                    if self._is_responding:
                        await self.client.cancel_response()
                        yield RealtimePipelineEvent(type="interrupted")
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            await self.client.close()

    async def _send_audio(self, audio_stream: AsyncIterator[bytes]):
        """Send audio chunks to the API"""
        async for chunk in audio_stream:
            await self.client.send_audio(chunk)

    def _convert_event(self, event: RealtimeEvent) -> RealtimePipelineEvent:
        """Convert API event to pipeline event"""
        mapping = {
            RealtimeEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                lambda e: RealtimePipelineEvent(type="speech_started"),
            RealtimeEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                lambda e: RealtimePipelineEvent(type="speech_ended"),
            RealtimeEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
                lambda e: RealtimePipelineEvent(type="transcript", data=e.transcript),
            RealtimeEventType.RESPONSE_AUDIO_DELTA:
                lambda e: RealtimePipelineEvent(type="audio", data=e.audio),
            RealtimeEventType.RESPONSE_DONE:
                lambda e: RealtimePipelineEvent(type="done"),
            RealtimeEventType.ERROR:
                lambda e: RealtimePipelineEvent(type="error", data=e.data)
        }

        converter = mapping.get(event.type)
        return converter(event) if converter else None
```

---

## Part 3: FastAPI Backend

### 3.1 Main Application

**File: `backend/main.py`**

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import base64

from src.pipelines.sandwich import SandwichPipeline
from src.pipelines.realtime import RealtimePipeline
from src.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("Voice Agents API starting...")
    yield
    print("Voice Agents API shutting down...")

app = FastAPI(
    title="Voice Agents API",
    description="Voice-enabled AI agents with Sandwich and Real-time architectures",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "voice-agents"}

@app.websocket("/ws/sandwich")
async def sandwich_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Sandwich architecture.

    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 audio>"}
    - Client sends: {"type": "end"} to signal end of speech
    - Server sends: {"type": "stt_partial", "text": "..."}
    - Server sends: {"type": "stt_final", "text": "..."}
    - Server sends: {"type": "agent_chunk", "text": "..."}
    - Server sends: {"type": "tts_audio", "data": "<base64 audio>"}
    - Server sends: {"type": "complete"}
    """
    await websocket.accept()

    pipeline = SandwichPipeline(
        elevenlabs_api_key=settings.ELEVENLABS_API_KEY,
        voice_id=settings.ELEVENLABS_VOICE_ID
    )

    audio_queue = asyncio.Queue()

    async def audio_generator():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    try:
        # Start pipeline processing task
        process_task = asyncio.create_task(
            process_sandwich_pipeline(websocket, pipeline, audio_generator())
        )

        # Receive audio from client
        while True:
            data = await websocket.receive_json()

            if data["type"] == "audio":
                audio_bytes = base64.b64decode(data["data"])
                await audio_queue.put(audio_bytes)
            elif data["type"] == "end":
                await audio_queue.put(None)
                break

        # Wait for processing to complete
        await process_task

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()

async def process_sandwich_pipeline(websocket, pipeline, audio_stream):
    """Process the sandwich pipeline and send events to client"""
    async for event in pipeline.process(audio_stream):
        if event.stage == "stt":
            await websocket.send_json({
                "type": f"stt_{event.event_type}",
                "text": event.data
            })
        elif event.stage == "agent":
            if event.event_type == "chunk":
                await websocket.send_json({
                    "type": "agent_chunk",
                    "text": event.data
                })
            elif event.event_type == "tool_call":
                await websocket.send_json({
                    "type": "agent_tool",
                    "tool": event.data
                })
        elif event.stage == "tts":
            if event.event_type == "audio_chunk" and event.data:
                await websocket.send_json({
                    "type": "tts_audio",
                    "data": base64.b64encode(event.data).decode()
                })

    await websocket.send_json({"type": "complete"})

@app.websocket("/ws/realtime")
async def realtime_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Real-time architecture.

    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 audio>"}
    - Server sends: {"type": "speech_started"}
    - Server sends: {"type": "speech_ended"}
    - Server sends: {"type": "transcript", "text": "..."}
    - Server sends: {"type": "audio", "data": "<base64 audio>"}
    - Server sends: {"type": "done"}
    """
    await websocket.accept()

    pipeline = RealtimePipeline(
        voice=settings.AZURE_REALTIME_VOICE,
        instructions=settings.AGENT_INSTRUCTIONS
    )

    audio_queue = asyncio.Queue()

    async def audio_generator():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    try:
        # Start processing in background
        process_task = asyncio.create_task(
            process_realtime_pipeline(websocket, pipeline, audio_generator())
        )

        # Receive audio from client
        while True:
            try:
                data = await websocket.receive_json()

                if data["type"] == "audio":
                    audio_bytes = base64.b64decode(data["data"])
                    await audio_queue.put(audio_bytes)
                elif data["type"] == "stop":
                    await audio_queue.put(None)
                    break
            except WebSocketDisconnect:
                break

        await process_task

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

async def process_realtime_pipeline(websocket, pipeline, audio_stream):
    """Process the realtime pipeline and send events to client"""
    async for event in pipeline.start(audio_stream):
        if event.type == "audio" and event.data:
            await websocket.send_json({
                "type": "audio",
                "data": base64.b64encode(event.data).decode()
            })
        elif event.type == "transcript":
            await websocket.send_json({
                "type": "transcript",
                "text": event.data
            })
        else:
            await websocket.send_json({"type": event.type})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3.2 Configuration

**File: `backend/config.py`**

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings from environment variables"""

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_REALTIME_DEPLOYMENT: str = "gpt-4o-realtime-preview"
    AZURE_REALTIME_VOICE: str = "alloy"

    # ElevenLabs
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # Agent
    AGENT_INSTRUCTIONS: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

**File: `backend/.env.example`**

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_REALTIME_DEPLOYMENT=gpt-4o-realtime-preview
AZURE_REALTIME_VOICE=alloy

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Optional Agent Instructions
AGENT_INSTRUCTIONS=You are a helpful voice assistant.
```

### 3.3 Requirements

**File: `backend/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
websockets==12.0
python-dotenv==1.0.1
pydantic-settings==2.2.1
langchain==0.3.0
langchain-openai==0.2.0
langgraph==0.2.0
httpx==0.27.0
```

---

## Part 4: Next.js Frontend

### 4.1 Project Configuration

**File: `frontend/package.json`**

```json
{
  "name": "voice-agents-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "zustand": "^4.5.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.4.0"
  }
}
```

### 4.2 Core Hooks

**File: `frontend/hooks/useAudioRecorder.ts`**

```typescript
import { useState, useRef, useCallback } from 'react';

interface UseAudioRecorderOptions {
  sampleRate?: number;
  onAudioChunk?: (chunk: ArrayBuffer) => void;
}

export function useAudioRecorder(options: UseAudioRecorderOptions = {}) {
  const { sampleRate = 16000, onAudioChunk } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      mediaStreamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext({ sampleRate });
      audioContextRef.current = audioContext;

      // Create source from stream
      const source = audioContext.createMediaStreamSource(stream);

      // Create processor for raw PCM data
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);

        // Convert Float32 to Int16 PCM
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        onAudioChunk?.(pcmData.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start recording');
    }
  }, [sampleRate, onAudioChunk]);

  const stopRecording = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    setIsRecording(false);
  }, []);

  return {
    isRecording,
    error,
    startRecording,
    stopRecording,
  };
}
```

**File: `frontend/hooks/useWebSocket.ts`**

```typescript
import { useState, useRef, useCallback, useEffect } from 'react';

interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onClose?: () => void;
}

export function useWebSocket(options: UseWebSocketOptions) {
  const { url, onMessage, onError, onClose } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    try {
      setError(null);
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          onMessage?.(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (event) => {
        setError('WebSocket error occurred');
        onError?.(event);
      };

      ws.onclose = () => {
        setIsConnected(false);
        onClose?.();
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect');
    }
  }, [url, onMessage, onError, onClose]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const send = useCallback((message: WebSocketMessage) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      // Convert to base64 for JSON transport
      const base64 = btoa(
        String.fromCharCode(...new Uint8Array(data))
      );
      wsRef.current.send(JSON.stringify({ type: 'audio', data: base64 }));
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    error,
    connect,
    disconnect,
    send,
    sendBinary,
  };
}
```

**File: `frontend/hooks/useAudioPlayer.ts`**

```typescript
import { useRef, useCallback, useState } from 'react';

export function useAudioPlayer(sampleRate: number = 16000) {
  const [isPlaying, setIsPlaying] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);

  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate });
    }
    return audioContextRef.current;
  }, [sampleRate]);

  const playChunk = useCallback((pcmData: ArrayBuffer) => {
    const int16Data = new Int16Array(pcmData);
    audioQueueRef.current.push(int16Data);

    if (!isPlayingRef.current) {
      playNextChunk();
    }
  }, []);

  const playNextChunk = useCallback(async () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      setIsPlaying(false);
      return;
    }

    isPlayingRef.current = true;
    setIsPlaying(true);

    const audioContext = initAudioContext();
    const int16Data = audioQueueRef.current.shift()!;

    // Convert Int16 to Float32
    const float32Data = new Float32Array(int16Data.length);
    for (let i = 0; i < int16Data.length; i++) {
      float32Data[i] = int16Data[i] / 32768;
    }

    // Create audio buffer
    const audioBuffer = audioContext.createBuffer(1, float32Data.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32Data);

    // Play buffer
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    source.onended = () => {
      playNextChunk();
    };

    source.start();
  }, [initAudioContext, sampleRate]);

  const stop = useCallback(() => {
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, []);

  return {
    isPlaying,
    playChunk,
    stop,
  };
}
```

### 4.3 Main Pages

**File: `frontend/app/page.tsx`**

```tsx
import Link from 'next/link';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold mb-4">Voice Agents</h1>
          <p className="text-xl text-slate-300">
            AI-powered voice interactions with two powerful architectures
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {/* Sandwich Mode Card */}
          <Link href="/sandwich" className="group">
            <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700 hover:border-blue-500 transition-all duration-300 h-full">
              <div className="text-4xl mb-4">🥪</div>
              <h2 className="text-2xl font-semibold mb-3 group-hover:text-blue-400 transition-colors">
                Sandwich Mode
              </h2>
              <p className="text-slate-400 mb-4">
                STT → Agent → TTS architecture using ElevenLabs for speech processing
                and Azure OpenAI for intelligent responses.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">ElevenLabs STT</span>
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">GPT-4o Agent</span>
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">ElevenLabs TTS</span>
              </div>
            </div>
          </Link>

          {/* Real-time Mode Card */}
          <Link href="/realtime" className="group">
            <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700 hover:border-green-500 transition-all duration-300 h-full">
              <div className="text-4xl mb-4">⚡</div>
              <h2 className="text-2xl font-semibold mb-3 group-hover:text-green-400 transition-colors">
                Real-time Mode
              </h2>
              <p className="text-slate-400 mb-4">
                Direct voice-to-voice using Azure OpenAI GPT-4o Realtime API with
                server-side VAD and integrated speech processing.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">GPT-4o Realtime</span>
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">Server VAD</span>
                <span className="px-3 py-1 bg-slate-700 rounded-full text-sm">Low Latency</span>
              </div>
            </div>
          </Link>
        </div>

        <div className="mt-16 text-center text-slate-500">
          <p>Built with Next.js, FastAPI, LangGraph, and Azure OpenAI</p>
        </div>
      </div>
    </main>
  );
}
```

**File: `frontend/app/sandwich/page.tsx`**

```tsx
'use client';

import { useState, useCallback } from 'react';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAudioPlayer } from '@/hooks/useAudioPlayer';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function SandwichPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [agentResponse, setAgentResponse] = useState('');
  const [status, setStatus] = useState<'idle' | 'recording' | 'processing' | 'speaking'>('idle');

  const audioPlayer = useAudioPlayer(16000);

  const handleMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'stt_partial':
        setCurrentTranscript(message.text);
        break;
      case 'stt_final':
        setCurrentTranscript(message.text);
        setMessages(prev => [...prev, { role: 'user', content: message.text }]);
        setStatus('processing');
        break;
      case 'agent_chunk':
        setAgentResponse(prev => prev + message.text);
        break;
      case 'tts_audio':
        setStatus('speaking');
        const audioData = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));
        audioPlayer.playChunk(audioData.buffer);
        break;
      case 'complete':
        setMessages(prev => [...prev, { role: 'assistant', content: agentResponse }]);
        setAgentResponse('');
        setCurrentTranscript('');
        setStatus('idle');
        break;
    }
  }, [agentResponse, audioPlayer]);

  const { isConnected, connect, disconnect, sendBinary, send } = useWebSocket({
    url: 'ws://localhost:8000/ws/sandwich',
    onMessage: handleMessage,
  });

  const handleAudioChunk = useCallback((chunk: ArrayBuffer) => {
    sendBinary(chunk);
  }, [sendBinary]);

  const { isRecording, startRecording, stopRecording } = useAudioRecorder({
    sampleRate: 16000,
    onAudioChunk: handleAudioChunk,
  });

  const handleStartRecording = async () => {
    connect();
    // Wait for connection
    await new Promise(resolve => setTimeout(resolve, 500));
    startRecording();
    setStatus('recording');
  };

  const handleStopRecording = () => {
    stopRecording();
    send({ type: 'end' });
    setStatus('processing');
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      <div className="container mx-auto px-4 py-8 max-w-3xl">
        <h1 className="text-3xl font-bold mb-2">Sandwich Mode</h1>
        <p className="text-slate-400 mb-8">STT → Agent → TTS Architecture</p>

        {/* Conversation History */}
        <div className="bg-slate-800 rounded-xl p-4 mb-6 h-96 overflow-y-auto">
          {messages.length === 0 ? (
            <p className="text-slate-500 text-center mt-32">
              Press and hold the button to speak
            </p>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-2 ${
                      msg.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700 text-slate-100'
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Live transcription */}
          {currentTranscript && (
            <div className="mt-4 p-3 bg-slate-700/50 rounded-lg border border-slate-600">
              <p className="text-sm text-slate-400 mb-1">Transcribing...</p>
              <p className="text-slate-200">{currentTranscript}</p>
            </div>
          )}

          {/* Live agent response */}
          {agentResponse && (
            <div className="mt-4 p-3 bg-green-900/30 rounded-lg border border-green-700">
              <p className="text-sm text-green-400 mb-1">Agent responding...</p>
              <p className="text-slate-200">{agentResponse}</p>
            </div>
          )}
        </div>

        {/* Status indicator */}
        <div className="text-center mb-6">
          <span className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${
            status === 'idle' ? 'bg-slate-700' :
            status === 'recording' ? 'bg-red-600' :
            status === 'processing' ? 'bg-yellow-600' :
            'bg-green-600'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              status === 'idle' ? 'bg-slate-400' : 'bg-white animate-pulse'
            }`} />
            {status === 'idle' && 'Ready'}
            {status === 'recording' && 'Recording...'}
            {status === 'processing' && 'Processing...'}
            {status === 'speaking' && 'Speaking...'}
          </span>
        </div>

        {/* Record button */}
        <div className="flex justify-center">
          <button
            onMouseDown={handleStartRecording}
            onMouseUp={handleStopRecording}
            onTouchStart={handleStartRecording}
            onTouchEnd={handleStopRecording}
            disabled={status === 'processing' || status === 'speaking'}
            className={`w-24 h-24 rounded-full flex items-center justify-center transition-all duration-200 ${
              isRecording
                ? 'bg-red-500 scale-110 shadow-lg shadow-red-500/50'
                : 'bg-blue-600 hover:bg-blue-500'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <svg
              className="w-10 h-10"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1 1.93c-3.94-.49-7-3.85-7-7.93h2c0 3.31 2.69 6 6 6s6-2.69 6-6h2c0 4.08-3.06 7.44-7 7.93V21h-2v-5.07z" />
            </svg>
          </button>
        </div>

        <p className="text-center text-slate-500 mt-4">
          Hold to speak, release to send
        </p>
      </div>
    </main>
  );
}
```

**File: `frontend/app/realtime/page.tsx`**

```tsx
'use client';

import { useState, useCallback, useRef } from 'react';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAudioPlayer } from '@/hooks/useAudioPlayer';

interface ConversationItem {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function RealtimePage() {
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [assistantSpeaking, setAssistantSpeaking] = useState(false);

  const transcriptBuffer = useRef('');
  const audioPlayer = useAudioPlayer(16000);

  const handleMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'speech_started':
        setUserSpeaking(true);
        audioPlayer.stop(); // Stop playback if user interrupts
        break;
      case 'speech_ended':
        setUserSpeaking(false);
        // Add user message to conversation
        if (transcriptBuffer.current) {
          setConversation(prev => [...prev, {
            role: 'user',
            content: transcriptBuffer.current,
            timestamp: new Date()
          }]);
          transcriptBuffer.current = '';
          setCurrentTranscript('');
        }
        break;
      case 'transcript':
        transcriptBuffer.current += message.text;
        setCurrentTranscript(transcriptBuffer.current);
        break;
      case 'audio':
        setAssistantSpeaking(true);
        const audioData = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));
        audioPlayer.playChunk(audioData.buffer);
        break;
      case 'done':
        setAssistantSpeaking(false);
        if (currentTranscript) {
          setConversation(prev => [...prev, {
            role: 'assistant',
            content: currentTranscript,
            timestamp: new Date()
          }]);
        }
        break;
      case 'interrupted':
        setAssistantSpeaking(false);
        audioPlayer.stop();
        break;
    }
  }, [audioPlayer, currentTranscript]);

  const { isConnected, connect, disconnect, sendBinary, send } = useWebSocket({
    url: 'ws://localhost:8000/ws/realtime',
    onMessage: handleMessage,
    onClose: () => {
      setIsSessionActive(false);
      setUserSpeaking(false);
      setAssistantSpeaking(false);
    }
  });

  const handleAudioChunk = useCallback((chunk: ArrayBuffer) => {
    sendBinary(chunk);
  }, [sendBinary]);

  const { isRecording, startRecording, stopRecording } = useAudioRecorder({
    sampleRate: 16000,
    onAudioChunk: handleAudioChunk,
  });

  const startSession = async () => {
    connect();
    await new Promise(resolve => setTimeout(resolve, 500));
    startRecording();
    setIsSessionActive(true);
  };

  const endSession = () => {
    stopRecording();
    send({ type: 'stop' });
    disconnect();
    setIsSessionActive(false);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      <div className="container mx-auto px-4 py-8 max-w-3xl">
        <h1 className="text-3xl font-bold mb-2">Real-time Mode</h1>
        <p className="text-slate-400 mb-8">Direct Voice-to-Voice with GPT-4o Realtime</p>

        {/* Conversation History */}
        <div className="bg-slate-800 rounded-xl p-4 mb-6 h-96 overflow-y-auto">
          {conversation.length === 0 && !isSessionActive ? (
            <p className="text-slate-500 text-center mt-32">
              Start a session to begin speaking
            </p>
          ) : (
            <div className="space-y-4">
              {conversation.map((item, idx) => (
                <div
                  key={idx}
                  className={`flex ${item.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-2 ${
                      item.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700 text-slate-100'
                    }`}
                  >
                    {item.content}
                  </div>
                </div>
              ))}

              {/* Live indicators */}
              {userSpeaking && (
                <div className="flex justify-end">
                  <div className="bg-blue-600/50 rounded-lg px-4 py-2 border border-blue-500">
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                      {currentTranscript || 'Listening...'}
                    </span>
                  </div>
                </div>
              )}

              {assistantSpeaking && (
                <div className="flex justify-start">
                  <div className="bg-green-600/50 rounded-lg px-4 py-2 border border-green-500">
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      Speaking...
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Status indicators */}
        <div className="flex justify-center gap-4 mb-6">
          <span className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            userSpeaking ? 'bg-blue-600' : 'bg-slate-700'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              userSpeaking ? 'bg-white animate-pulse' : 'bg-slate-500'
            }`} />
            You
          </span>
          <span className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            assistantSpeaking ? 'bg-green-600' : 'bg-slate-700'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              assistantSpeaking ? 'bg-white animate-pulse' : 'bg-slate-500'
            }`} />
            Assistant
          </span>
        </div>

        {/* Session control */}
        <div className="flex justify-center">
          {!isSessionActive ? (
            <button
              onClick={startSession}
              className="px-8 py-4 bg-green-600 hover:bg-green-500 rounded-full text-lg font-semibold transition-colors"
            >
              Start Conversation
            </button>
          ) : (
            <button
              onClick={endSession}
              className="px-8 py-4 bg-red-600 hover:bg-red-500 rounded-full text-lg font-semibold transition-colors"
            >
              End Conversation
            </button>
          )}
        </div>

        {isSessionActive && (
          <p className="text-center text-slate-500 mt-4">
            Just speak naturally - the AI will respond automatically
          </p>
        )}
      </div>
    </main>
  );
}
```

### 4.4 Layout and Styling

**File: `frontend/app/layout.tsx`**

```tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Voice Agents',
  description: 'AI-powered voice interactions with Sandwich and Real-time architectures',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
```

**File: `frontend/app/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --foreground-rgb: 255, 255, 255;
  --background-start-rgb: 15, 23, 42;
  --background-end-rgb: 30, 41, 59;
}

body {
  color: rgb(var(--foreground-rgb));
  background: linear-gradient(
    to bottom right,
    rgb(var(--background-start-rgb)),
    rgb(var(--background-end-rgb))
  );
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: rgb(30, 41, 59);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgb(71, 85, 105);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgb(100, 116, 139);
}
```

---

## Part 5: Setup and Deployment

### 5.1 Backend Setup

```bash
# Navigate to backend directory
cd VoiceAgents/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and fill in values
cp .env.example .env

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5.2 Frontend Setup

```bash
# Navigate to frontend directory
cd VoiceAgents/frontend

# Install dependencies
npm install

# Copy environment template
cp .env.local.example .env.local

# Run development server
npm run dev
```

### 5.3 Environment Variables

**Backend (.env):**
| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_DEPLOYMENT` | GPT-4o deployment name | Yes |
| `AZURE_OPENAI_REALTIME_DEPLOYMENT` | GPT-4o-realtime deployment | Yes |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Yes |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID | No (default: Rachel) |

**Frontend (.env.local):**
| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes |

---

## Part 6: Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure
- [ ] Configure backend with FastAPI
- [ ] Configure frontend with Next.js
- [ ] Implement audio recording hooks
- [ ] Implement WebSocket communication

### Phase 2: Sandwich Architecture (Week 2)
- [ ] Integrate ElevenLabs STT
- [ ] Build LangGraph agent with Azure OpenAI
- [ ] Integrate ElevenLabs TTS
- [ ] Implement sandwich pipeline orchestration
- [ ] Build sandwich mode UI

### Phase 3: Real-time Architecture (Week 3)
- [ ] Integrate Azure OpenAI Realtime API
- [ ] Implement server VAD handling
- [ ] Build real-time pipeline
- [ ] Build real-time mode UI
- [ ] Add interruption handling

### Phase 4: Polish & Testing (Week 4)
- [ ] Add error handling and recovery
- [ ] Implement conversation persistence
- [ ] Add loading states and animations
- [ ] Performance optimization
- [ ] End-to-end testing

---

## Appendix: API Reference

### WebSocket Protocol: Sandwich Mode

**Client → Server:**
```json
{"type": "audio", "data": "<base64 PCM audio>"}
{"type": "end"}
```

**Server → Client:**
```json
{"type": "stt_partial", "text": "transcription in progress..."}
{"type": "stt_final", "text": "final transcription"}
{"type": "agent_chunk", "text": "response token"}
{"type": "agent_tool", "tool": {"name": "...", "args": {}}}
{"type": "tts_audio", "data": "<base64 PCM audio>"}
{"type": "complete"}
{"type": "error", "message": "error description"}
```

### WebSocket Protocol: Real-time Mode

**Client → Server:**
```json
{"type": "audio", "data": "<base64 PCM audio>"}
{"type": "stop"}
```

**Server → Client:**
```json
{"type": "speech_started"}
{"type": "speech_ended"}
{"type": "transcript", "text": "transcribed text"}
{"type": "audio", "data": "<base64 PCM audio>"}
{"type": "done"}
{"type": "interrupted"}
{"type": "error", "message": "error description"}
```

---

## Appendix: Troubleshooting

### Common Issues

1. **Microphone not working**
   - Ensure browser has microphone permissions
   - Check if another application is using the microphone
   - Verify sample rate compatibility (16kHz recommended)

2. **WebSocket connection failing**
   - Verify backend is running on correct port
   - Check CORS configuration
   - Verify firewall settings

3. **Audio playback issues**
   - Ensure AudioContext is initialized after user interaction
   - Check for sample rate mismatch
   - Verify audio data is properly decoded

4. **Azure Realtime API errors**
   - Verify deployment is in supported region (East US 2 or Sweden Central)
   - Check API version (2025-04-01-preview)
   - Verify model deployment name

5. **ElevenLabs API errors**
   - Verify API key is valid
   - Check voice ID exists
   - Monitor API rate limits
