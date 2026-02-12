"""
Voice-Based System Demos using Streamlit
=========================================
Demo 1: Text to Audio (non-streaming) - ElevenLabs
Demo 2: Text to Audio (streaming) - ElevenLabs
Demo 3: Audio file to Text transcription
Demo 4: Real-time Speech to Text streaming
"""

import os
import base64
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from io import BytesIO

from audio_recorder_streamlit import audio_recorder

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / "backend" / ".env")

# ElevenLabs imports
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Initialize ElevenLabs client
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Create output directory for audio files
OUTPUT_DIR = Path(__file__).parent / "audio_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Voice Demos",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Voice-Based System Demos")

# Sidebar for demo selection
demo_choice = st.sidebar.selectbox(
    "Select Demo",
    [
        "Demo 1: Text to Audio (Non-Streaming)",
        "Demo 2: Text to Audio (Streaming)",
        "Demo 3: Audio to Text Transcription",
        "Demo 4: Real-time Speech to Text"
    ]
)

# ============================================================================
# DEMO 1: Text to Audio (Non-Streaming)
# ============================================================================
def demo_text_to_audio_nonstreaming():
    """
    Convert text to audio using ElevenLabs API (non-streaming).
    The entire audio is generated first, then saved to file and played.
    """
    st.header("Demo 1: Text to Audio (Non-Streaming)")
    st.markdown("""
    This demo converts text to speech using ElevenLabs API.
    The audio is generated completely before being saved and played.
    """)

    # Text input
    text_input = st.text_area(
        "Enter text to convert to speech:",
        value="Hello! This is a demonstration of text to speech conversion using ElevenLabs.",
        height=150
    )

    # Voice settings
    col1, col2 = st.columns(2)
    with col1:
        stability = st.slider("Stability", 0.0, 1.0, 0.5, 0.1)
        similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, 0.75, 0.1)
    with col2:
        style = st.slider("Style", 0.0, 1.0, 0.0, 0.1)
        speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)

    # Model selection
    model_id = st.selectbox(
        "Select Model",
        ["eleven_multilingual_v2", "eleven_monolingual_v1", "eleven_turbo_v2_5"],
        index=0
    )

    if st.button("🔊 Generate Audio", key="generate_nonstream"):
        if not text_input.strip():
            st.error("Please enter some text to convert.")
            return

        if not ELEVENLABS_API_KEY:
            st.error("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in .env file.")
            return

        with st.spinner("Generating audio..."):
            try:
                # Generate audio using ElevenLabs
                audio = elevenlabs_client.text_to_speech.convert(
                    text=text_input,
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id=model_id,
                    output_format="mp3_44100_128",
                    voice_settings=VoiceSettings(
                        stability=stability,
                        similarity_boost=similarity_boost,
                        style=style,
                        use_speaker_boost=True,
                        speed=speed,
                    ),
                )

                # Convert generator to bytes
                audio_bytes = b"".join(audio)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tts_nonstream_{timestamp}.mp3"
                filepath = OUTPUT_DIR / filename

                # Save to file
                with open(filepath, "wb") as f:
                    f.write(audio_bytes)

                st.success(f"✅ Audio generated and saved to: `{filepath}`")

                # Display audio player
                st.audio(audio_bytes, format="audio/mp3")

                # Download button
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/mp3"
                )

            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")


# ============================================================================
# DEMO 2: Text to Audio (Streaming with Real-time Playback)
# ============================================================================

def create_audio_player_html(audio_chunks_b64: list[str]) -> str:
    """
    Create an HTML page with JavaScript that plays audio chunks sequentially.
    All chunks are embedded in the HTML and played via Web Audio API.
    """
    chunks_js_array = ",".join([f'"{chunk}"' for chunk in audio_chunks_b64])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 10px;
                margin: 0;
                background: transparent;
            }}
            .player-status {{
                padding: 8px 12px;
                border-radius: 4px;
                background: #e8f4e8;
                color: #2e7d32;
                font-size: 14px;
            }}
            .player-status.playing {{
                background: #e3f2fd;
                color: #1565c0;
            }}
        </style>
    </head>
    <body>
        <div id="status" class="player-status">Initializing audio player...</div>

        <script>
            const audioChunks = [{chunks_js_array}];
            let audioContext = null;
            let currentIndex = 0;
            let isPlaying = false;

            function updateStatus(text, isPlaying) {{
                const status = document.getElementById('status');
                status.textContent = text;
                status.className = 'player-status' + (isPlaying ? ' playing' : '');
            }}

            async function initAudio() {{
                if (!audioContext) {{
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }}
                if (audioContext.state === 'suspended') {{
                    await audioContext.resume();
                }}
            }}

            async function decodeChunk(base64Data) {{
                const binaryString = atob(base64Data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                try {{
                    return await audioContext.decodeAudioData(bytes.buffer.slice(0));
                }} catch (err) {{
                    console.log('Decode error:', err);
                    return null;
                }}
            }}

            async function playNext() {{
                if (currentIndex >= audioChunks.length) {{
                    updateStatus('✅ Playback complete!', false);
                    return;
                }}

                updateStatus(`🎵 Playing chunk ${{currentIndex + 1}} of ${{audioChunks.length}}...`, true);

                const buffer = await decodeChunk(audioChunks[currentIndex]);
                currentIndex++;

                if (!buffer) {{
                    playNext();
                    return;
                }}

                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.onended = () => playNext();
                source.start(0);
            }}

            async function startPlayback() {{
                await initAudio();
                if (audioChunks.length === 0) {{
                    updateStatus('No audio chunks to play', false);
                    return;
                }}
                updateStatus(`🎵 Starting playback (${{audioChunks.length}} chunks)...`, true);
                playNext();
            }}

            // Auto-start playback
            startPlayback();
        </script>
    </body>
    </html>
    """


def demo_text_to_audio_streaming():
    """
    Convert text to audio using ElevenLabs API (streaming).
    Audio chunks are collected and then played sequentially via Web Audio API.
    The complete audio is saved to file at the end.
    """
    st.header("Demo 2: Text to Audio (Streaming)")
    st.markdown("""
    This demo converts text to speech using ElevenLabs **streaming** API.
    - Audio chunks are streamed from the API in real-time
    - Chunks are played sequentially in your browser
    - Complete audio is saved to file after streaming finishes
    """)

    # Text input
    text_input = st.text_area(
        "Enter text to convert to speech:",
        value="This is a streaming demonstration. The audio is generated chunk by chunk and played in real-time. This approach provides low latency for voice applications.",
        height=150,
        key="streaming_text"
    )

    # Voice settings
    col1, col2 = st.columns(2)
    with col1:
        stability = st.slider("Stability", 0.0, 1.0, 0.5, 0.1, key="stream_stability")
        similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, 0.75, 0.1, key="stream_similarity")
    with col2:
        style = st.slider("Style", 0.0, 1.0, 0.0, 0.1, key="stream_style")
        speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1, key="stream_speed")

    # Model selection
    model_id = st.selectbox(
        "Select Model",
        ["eleven_multilingual_v2", "eleven_monolingual_v1", "eleven_turbo_v2_5"],
        index=0,
        key="stream_model"
    )

    if st.button("🔊 Generate & Stream Audio", key="generate_stream"):
        if not text_input.strip():
            st.error("Please enter some text to convert.")
            return

        if not ELEVENLABS_API_KEY:
            st.error("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in .env file.")
            return

        # Create placeholders for real-time updates
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        chunk_info = st.empty()

        try:
            status_placeholder.info("🎵 Streaming audio from ElevenLabs...")

            # Use streaming API
            response = elevenlabs_client.text_to_speech.stream(
                voice_id=ELEVENLABS_VOICE_ID,
                output_format="mp3_44100_128",
                text=text_input,
                model_id=model_id,
                voice_settings=VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
                    use_speaker_boost=True,
                    speed=speed,
                ),
            )

            # Collect chunks
            audio_chunks = []
            audio_chunks_b64 = []
            chunk_count = 0
            total_bytes = 0
            estimated_chunks = max(10, len(text_input) // 15)

            # Buffer to accumulate enough data for valid MP3 frames
            chunk_buffer = b""
            min_chunk_size = 8000  # ~8KB for reliable MP3 decode

            for chunk in response:
                if chunk:
                    audio_chunks.append(chunk)
                    chunk_buffer += chunk
                    chunk_count += 1
                    total_bytes += len(chunk)

                    # Update progress
                    progress = min(chunk_count / estimated_chunks, 0.99)
                    progress_bar.progress(progress)
                    chunk_info.markdown(f"📦 Receiving chunk **{chunk_count}** | Total: **{total_bytes:,}** bytes")

                    # Store chunk for playback when buffer is large enough
                    if len(chunk_buffer) >= min_chunk_size:
                        chunk_b64 = base64.b64encode(chunk_buffer).decode('utf-8')
                        audio_chunks_b64.append(chunk_b64)
                        chunk_buffer = b""

            # Add any remaining buffered audio
            if chunk_buffer:
                chunk_b64 = base64.b64encode(chunk_buffer).decode('utf-8')
                audio_chunks_b64.append(chunk_b64)

            # Complete progress
            progress_bar.progress(1.0)
            status_placeholder.success(f"✅ Received {chunk_count} chunks ({total_bytes:,} bytes). Starting playback...")

            # Create and display the audio player
            player_html = create_audio_player_html(audio_chunks_b64)
            components.html(player_html, height=60)

            # Combine all chunks for final audio
            audio_bytes = b"".join(audio_chunks)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_stream_{timestamp}.mp3"
            filepath = OUTPUT_DIR / filename

            # Save to file
            with open(filepath, "wb") as f:
                f.write(audio_bytes)

            chunk_info.markdown(f"📦 Total: **{chunk_count}** chunks | **{total_bytes:,}** bytes | **{len(audio_chunks_b64)}** playable segments")

            # Show complete audio for replay
            st.markdown("---")
            st.markdown(f"**Audio saved to:** `{filepath}`")
            st.markdown("**Complete Audio (for replay):**")
            st.audio(audio_bytes, format="audio/mp3")

            # Download button
            st.download_button(
                label="📥 Download Audio",
                data=audio_bytes,
                file_name=filename,
                mime="audio/mp3",
                key="download_stream"
            )

        except Exception as e:
            status_placeholder.error(f"Error during streaming: {str(e)}")
            progress_bar.empty()


# ============================================================================
# DEMO 3: Audio to Text Transcription
# ============================================================================
def demo_audio_to_text():
    """
    Record audio from microphone or upload a file, then transcribe using ElevenLabs.
    """
    st.header("Demo 3: Audio to Text Transcription")
    st.markdown("""
    This demo transcribes audio to text using ElevenLabs **Speech-to-Text** API.
    - Record audio from your microphone or upload an audio file
    - Supports speaker diarization (who is speaking)
    - Tags audio events like laughter, applause, etc.
    """)

    # Choose input method
    input_method = st.radio(
        "Select audio input method:",
        ["🎤 Record from Microphone", "📁 Upload Audio File"],
        horizontal=True,
        key="stt_input_method"
    )

    audio_data = None
    audio_bytes = None

    if input_method == "🎤 Record from Microphone":
        st.markdown("### Record Audio")
        st.info("Click the microphone button below to start recording. Click again to stop.")

        # Audio recorder widget
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x",
            pause_threshold=3.0,  # Stop after 3 seconds of silence
        )

        if audio_bytes:
            st.success(f"✅ Recording captured! Size: {len(audio_bytes):,} bytes")
            st.audio(audio_bytes, format="audio/wav")
            audio_data = BytesIO(audio_bytes)

    else:  # Upload file
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
            key="stt_file_upload"
        )

        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.success(f"✅ File uploaded: {uploaded_file.name} ({len(audio_bytes):,} bytes)")
            st.audio(audio_bytes, format=f"audio/{uploaded_file.type.split('/')[-1]}")
            audio_data = BytesIO(audio_bytes)

    # Transcription settings
    st.markdown("---")
    st.markdown("### Transcription Settings")

    col1, col2 = st.columns(2)
    with col1:
        language_code = st.selectbox(
            "Language",
            [
                ("Auto-detect", None),
                ("English", "eng"),
                ("Spanish", "spa"),
                ("French", "fra"),
                ("German", "deu"),
                ("Italian", "ita"),
                ("Portuguese", "por"),
                ("Dutch", "nld"),
                ("Polish", "pol"),
                ("Russian", "rus"),
                ("Japanese", "jpn"),
                ("Chinese", "cmn"),
                ("Korean", "kor"),
                ("Arabic", "ara"),
                ("Hindi", "hin"),
            ],
            format_func=lambda x: x[0],
            key="stt_language"
        )

    with col2:
        diarize = st.checkbox("Enable Speaker Diarization", value=True, key="stt_diarize",
                              help="Identify and label different speakers")
        tag_events = st.checkbox("Tag Audio Events", value=True, key="stt_tag_events",
                                 help="Tag events like laughter, applause, music, etc.")

    # Transcribe button
    if st.button("🎯 Transcribe Audio", key="transcribe_btn", disabled=audio_data is None):
        if not ELEVENLABS_API_KEY:
            st.error("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in .env file.")
            return

        with st.spinner("Transcribing audio..."):
            try:
                # Reset the BytesIO position
                audio_data.seek(0)

                # Call ElevenLabs Speech-to-Text API
                transcription = elevenlabs_client.speech_to_text.convert(
                    file=audio_data,
                    model_id="scribe_v1",
                    tag_audio_events=tag_events,
                    language_code=language_code[1],  # Get the code, not the display name
                    diarize=diarize,
                )

                st.success("✅ Transcription complete!")

                # Display results
                st.markdown("---")
                st.markdown("### Transcription Result")

                # Show the full text
                if hasattr(transcription, 'text'):
                    st.markdown("**Full Text:**")
                    st.text_area("Transcription", transcription.text, height=200, key="transcription_result")

                    # Copy button
                    st.download_button(
                        label="📋 Download Transcription",
                        data=transcription.text,
                        file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_transcription"
                    )

                # Show detailed info if available
                if hasattr(transcription, 'words') and transcription.words:
                    with st.expander("📝 Word-level Details"):
                        for word in transcription.words[:50]:  # Show first 50 words
                            st.write(f"- **{word.text}** (start: {word.start:.2f}s, end: {word.end:.2f}s)")
                        if len(transcription.words) > 50:
                            st.write(f"... and {len(transcription.words) - 50} more words")

                # Show speaker segments if diarization was enabled
                if diarize and hasattr(transcription, 'utterances') and transcription.utterances:
                    with st.expander("🗣️ Speaker Segments"):
                        for utterance in transcription.utterances:
                            speaker = getattr(utterance, 'speaker', 'Unknown')
                            text = getattr(utterance, 'text', '')
                            st.markdown(f"**Speaker {speaker}:** {text}")

                # Show audio events if tagged
                if tag_events and hasattr(transcription, 'audio_events') and transcription.audio_events:
                    with st.expander("🎭 Audio Events"):
                        for event in transcription.audio_events:
                            event_type = getattr(event, 'type', 'Unknown')
                            start = getattr(event, 'start', 0)
                            end = getattr(event, 'end', 0)
                            st.write(f"- **{event_type}** ({start:.2f}s - {end:.2f}s)")

                # Show raw response for debugging
                with st.expander("🔍 Raw API Response"):
                    st.json(transcription.model_dump() if hasattr(transcription, 'model_dump') else str(transcription))

            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")

    # Show placeholder if no audio
    if audio_data is None:
        st.info("👆 Record or upload audio to transcribe")


# ============================================================================
# DEMO 4: Real-time Speech to Text Streaming
# ============================================================================

def create_browser_realtime_stt_html(api_key: str) -> str:
    """
    Create a fully client-side real-time STT implementation.
    JavaScript handles microphone capture and WebSocket to ElevenLabs directly.
    This bypasses Streamlit's Python backend entirely for the real-time part.
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 20px;
                margin: 0;
                background: #fafafa;
                min-height: 400px;
            }}
            .container {{
                max-width: 100%;
            }}
            .controls {{
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }}
            button {{
                padding: 12px 28px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 15px;
                font-weight: 600;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            button:hover:not(:disabled) {{
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            button:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            #startBtn {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
            }}
            #stopBtn {{
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
            }}
            .status-badge {{
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            .status-badge.idle {{
                background: #f3f4f6;
                color: #6b7280;
            }}
            .status-badge.connecting {{
                background: #fef3c7;
                color: #d97706;
            }}
            .status-badge.recording {{
                background: #fee2e2;
                color: #dc2626;
                animation: pulse 1.5s infinite;
            }}
            .status-badge.connected {{
                background: #d1fae5;
                color: #059669;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
            }}
            .visualizer {{
                height: 50px;
                background: white;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
                padding: 0 20px;
                border: 1px solid #e5e7eb;
            }}
            .bar {{
                width: 5px;
                height: 8px;
                background: #10b981;
                border-radius: 3px;
                transition: height 0.05s ease-out;
            }}
            .transcript-box {{
                background: white;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 15px;
                border: 1px solid #e5e7eb;
                min-height: 60px;
            }}
            .transcript-label {{
                font-size: 12px;
                font-weight: 600;
                color: #6b7280;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .partial-text {{
                color: #9ca3af;
                font-style: italic;
                font-size: 15px;
                min-height: 24px;
            }}
            .final-text {{
                color: #1f2937;
                font-size: 15px;
                line-height: 1.6;
                white-space: pre-wrap;
                max-height: 200px;
                overflow-y: auto;
            }}
            .error-box {{
                background: #fef2f2;
                border: 1px solid #fecaca;
                color: #dc2626;
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 15px;
                font-size: 14px;
            }}
            .info-text {{
                font-size: 13px;
                color: #6b7280;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="controls">
                <button id="startBtn" onclick="startTranscription()">
                    <span>Start</span>
                </button>
                <button id="stopBtn" onclick="stopTranscription()" disabled>
                    <span>Stop</span>
                </button>
                <span id="status" class="status-badge idle">Ready</span>
            </div>

            <div class="visualizer" id="visualizer">
                {"".join(['<div class="bar"></div>' for _ in range(20)])}
            </div>

            <div id="errorBox" class="error-box" style="display: none;"></div>

            <div class="transcript-box">
                <div class="transcript-label">Live (partial)</div>
                <div id="partialText" class="partial-text">Waiting for speech...</div>
            </div>

            <div class="transcript-box">
                <div class="transcript-label">Transcribed Text</div>
                <div id="finalText" class="final-text"></div>
            </div>

            <div class="info-text">
                Tip: Speak clearly. Transcription commits automatically after pauses.
            </div>
        </div>

        <script>
            const API_KEY = "{api_key}";
            const WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime";
            const SAMPLE_RATE = 16000;

            let websocket = null;
            let audioContext = null;
            let mediaStream = null;
            let processor = null;
            let isRunning = false;

            function updateStatus(text, className) {{
                const status = document.getElementById('status');
                status.textContent = text;
                status.className = 'status-badge ' + className;
            }}

            function showError(message) {{
                const errorBox = document.getElementById('errorBox');
                errorBox.textContent = message;
                errorBox.style.display = 'block';
            }}

            function hideError() {{
                document.getElementById('errorBox').style.display = 'none';
            }}

            function updateVisualizer(level) {{
                const bars = document.querySelectorAll('.bar');
                bars.forEach((bar, i) => {{
                    const variation = 0.4 + Math.random() * 0.6;
                    const height = Math.max(8, Math.min(45, level * variation * 500));
                    bar.style.height = height + 'px';
                    bar.style.background = level > 0.01 ? '#10b981' : '#d1d5db';
                }});
            }}

            function resetVisualizer() {{
                document.querySelectorAll('.bar').forEach(bar => {{
                    bar.style.height = '8px';
                    bar.style.background = '#d1d5db';
                }});
            }}

            async function startTranscription() {{
                hideError();
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                updateStatus('Connecting...', 'connecting');

                try {{
                    // Connect WebSocket to ElevenLabs
                    const wsUrl = `${{WS_URL}}?model_id=scribe_v2_realtime&sample_rate=${{SAMPLE_RATE}}&language_code=en`;
                    websocket = new WebSocket(wsUrl);

                    websocket.onopen = async () => {{
                        console.log('WebSocket connected');
                        // Send API key as first message
                        websocket.send(JSON.stringify({{
                            "type": "auth",
                            "xi_api_key": API_KEY
                        }}));

                        // Start microphone capture
                        await startMicrophone();
                        updateStatus('Recording...', 'recording');
                        isRunning = true;
                    }};

                    websocket.onmessage = (event) => {{
                        try {{
                            const data = JSON.parse(event.data);
                            console.log('WS message:', data.type);

                            if (data.type === 'partial_transcript') {{
                                document.getElementById('partialText').textContent =
                                    data.text || 'Listening...';
                            }} else if (data.type === 'final_transcript' ||
                                       data.type === 'committed_transcript') {{
                                const finalDiv = document.getElementById('finalText');
                                if (data.text && data.text.trim()) {{
                                    finalDiv.textContent += (finalDiv.textContent ? ' ' : '') + data.text;
                                    finalDiv.scrollTop = finalDiv.scrollHeight;
                                }}
                                document.getElementById('partialText').textContent = 'Listening...';
                            }} else if (data.type === 'error') {{
                                showError(data.message || 'Transcription error');
                            }}
                        }} catch (e) {{
                            console.log('Parse error:', e);
                        }}
                    }};

                    websocket.onerror = (error) => {{
                        console.error('WebSocket error:', error);
                        showError('Connection error. Check your API key.');
                        stopTranscription();
                    }};

                    websocket.onclose = (event) => {{
                        console.log('WebSocket closed:', event.code, event.reason);
                        if (isRunning) {{
                            updateStatus('Disconnected', 'idle');
                        }}
                    }};

                }} catch (error) {{
                    console.error('Error:', error);
                    showError('Failed to start: ' + error.message);
                    stopTranscription();
                }}
            }}

            async function startMicrophone() {{
                try {{
                    mediaStream = await navigator.mediaDevices.getUserMedia({{
                        audio: {{
                            sampleRate: SAMPLE_RATE,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true,
                        }}
                    }});

                    audioContext = new (window.AudioContext || window.webkitAudioContext)({{
                        sampleRate: SAMPLE_RATE
                    }});

                    const source = audioContext.createMediaStreamSource(mediaStream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);

                    processor.onaudioprocess = (e) => {{
                        if (!isRunning || !websocket || websocket.readyState !== WebSocket.OPEN) return;

                        const inputData = e.inputBuffer.getChannelData(0);

                        // Calculate level for visualizer
                        let sum = 0;
                        for (let i = 0; i < inputData.length; i++) {{
                            sum += Math.abs(inputData[i]);
                        }}
                        updateVisualizer(sum / inputData.length);

                        // Convert to 16-bit PCM
                        const pcmData = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {{
                            const s = Math.max(-1, Math.min(1, inputData[i]));
                            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        }}

                        // Convert to base64
                        const bytes = new Uint8Array(pcmData.buffer);
                        let binary = '';
                        for (let i = 0; i < bytes.length; i++) {{
                            binary += String.fromCharCode(bytes[i]);
                        }}
                        const base64 = btoa(binary);

                        // Send to WebSocket
                        websocket.send(JSON.stringify({{
                            "audio_base_64": base64
                        }}));
                    }};

                    source.connect(processor);
                    processor.connect(audioContext.destination);

                }} catch (error) {{
                    throw new Error('Microphone access denied: ' + error.message);
                }}
            }}

            function stopTranscription() {{
                isRunning = false;

                // Close WebSocket
                if (websocket) {{
                    try {{
                        // Send commit before closing
                        if (websocket.readyState === WebSocket.OPEN) {{
                            websocket.send(JSON.stringify({{ "commit": true }}));
                        }}
                        websocket.close();
                    }} catch (e) {{}}
                    websocket = null;
                }}

                // Stop audio
                if (processor) {{
                    processor.disconnect();
                    processor = null;
                }}
                if (mediaStream) {{
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }}
                if (audioContext) {{
                    audioContext.close();
                    audioContext = null;
                }}

                // Update UI
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                updateStatus('Stopped', 'idle');
                document.getElementById('partialText').textContent = 'Click Start to begin';
                resetVisualizer();
            }}

            // Cleanup on page unload
            window.addEventListener('beforeunload', stopTranscription);
        </script>
    </body>
    </html>
    """


def demo_realtime_stt():
    """
    Real-time speech to text using ElevenLabs WebSocket API.
    Uses a fully client-side JavaScript implementation for reliable real-time transcription.
    """
    st.header("Demo 4: Real-time Speech to Text")
    st.markdown("""
    This demo provides **real-time speech transcription** using ElevenLabs Scribe API.
    - Audio is captured directly in your browser
    - Streamed to ElevenLabs via WebSocket for instant transcription
    - See partial transcripts update as you speak
    - Fully client-side - no Python WebRTC issues
    """)

    if not ELEVENLABS_API_KEY:
        st.error("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in .env file.")
        return

    st.markdown("---")
    st.markdown("### Real-time Transcription")
    st.info("Click **Start** to begin. Allow microphone access when prompted.")

    # Embed the fully client-side real-time STT component
    realtime_html = create_browser_realtime_stt_html(ELEVENLABS_API_KEY)
    components.html(realtime_html, height=550, scrolling=False)

    # Instructions
    with st.expander("How it works"):
        st.markdown("""
        **Real-time Transcription Flow:**

        1. Click **Start** to connect to ElevenLabs
        2. Allow microphone access in your browser
        3. Speak into your microphone
        4. Watch partial transcripts appear as you speak
        5. Final transcripts commit automatically after pauses
        6. Click **Stop** when done

        **Technical Details:**
        - Audio captured at 16kHz mono PCM
        - Streamed via WebSocket to ElevenLabs Scribe v2
        - Uses browser's Web Audio API for processing
        - No Python backend involved for real-time part
        """)


# ============================================================================
# Main App Logic
# ============================================================================
if demo_choice == "Demo 1: Text to Audio (Non-Streaming)":
    demo_text_to_audio_nonstreaming()
elif demo_choice == "Demo 2: Text to Audio (Streaming)":
    demo_text_to_audio_streaming()
elif demo_choice == "Demo 3: Audio to Text Transcription":
    demo_audio_to_text()
elif demo_choice == "Demo 4: Real-time Speech to Text":
    demo_realtime_stt()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.markdown(f"**Voice ID:** `{ELEVENLABS_VOICE_ID[:8]}...`")
st.sidebar.markdown(f"**Output Dir:** `{OUTPUT_DIR}`")
