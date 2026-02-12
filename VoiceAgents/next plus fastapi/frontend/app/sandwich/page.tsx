"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { useAudioPlayer } from "@/hooks/useAudioPlayer";
import { useWebSocket } from "@/hooks/useWebSocket";
import { base64ToArrayBuffer } from "@/lib/audio-utils";
import { getWebSocketUrl } from "@/lib/websocket";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function SandwichPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState("");
  const [agentResponse, setAgentResponse] = useState("");
  const [audioStopped, setAudioStopped] = useState(false);
  const [status, setStatus] = useState<"idle" | "recording" | "processing" | "speaking">("idle");

  const hasTtsRef = useRef(false);
  const assistantCommittedRef = useRef(false);
  const audioPlayer = useAudioPlayer(16000, () => {
    if (hasTtsRef.current) {
      hasTtsRef.current = false;
      setStatus("idle");
      if (agentBufferRef.current && !assistantCommittedRef.current) {
        setMessages((prev) => [...prev, { role: "assistant", content: agentBufferRef.current }]);
        assistantCommittedRef.current = true;
      }
    }
  });
  const agentBufferRef = useRef("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const { connect, disconnect, sendBinary, send } = useWebSocket({
    url: getWebSocketUrl("/ws/sandwich"),
    onMessage: (message: any) => {
      switch (message.type) {
        case "stt_partial":
          setCurrentTranscript(message.text);
          break;
        case "stt_final":
          setMessages((prev) => [...prev, { role: "user", content: message.text }]);
          setCurrentTranscript("");
          agentBufferRef.current = "";
          assistantCommittedRef.current = false;
          setAgentResponse("");
          setAudioStopped(false);
          hasTtsRef.current = false;
          setStatus("processing");
          break;
        case "agent_chunk":
          agentBufferRef.current += message.text;
          setAgentResponse(agentBufferRef.current);
          break;
        case "tts_audio":
          setStatus("speaking");
          if (agentBufferRef.current && !assistantCommittedRef.current) {
            setMessages((prev) => [...prev, { role: "assistant", content: agentBufferRef.current }]);
            assistantCommittedRef.current = true;
          }
          if (!audioStopped) {
            hasTtsRef.current = true;
            audioPlayer.playChunk(base64ToArrayBuffer(message.data));
          }
          break;
        case "complete":
          if (agentBufferRef.current && !assistantCommittedRef.current) {
            setMessages((prev) => [...prev, { role: "assistant", content: agentBufferRef.current }]);
            assistantCommittedRef.current = true;
          }
          agentBufferRef.current = "";
          setAgentResponse("");
          if (!hasTtsRef.current) {
            setStatus("idle");
          }
          disconnect();
          break;
        case "error":
          setStatus("idle");
          break;
        default:
          break;
      }
    },
  });

  const handleAudioChunk = useCallback(
    (chunk: ArrayBuffer) => {
      sendBinary(chunk);
    },
    [sendBinary]
  );

  const { isRecording, startRecording, stopRecording } = useAudioRecorder({
    sampleRate: 16000,
    onAudioChunk: handleAudioChunk,
  });

  const handleStartRecording = async () => {
    connect();
    await new Promise((resolve) => setTimeout(resolve, 300));
    if (messages.length) {
      send({ type: "history", messages });
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
    startRecording();
    setStatus("recording");
  };

  const handleStopRecording = () => {
    stopRecording();
    send({ type: "end" });
    setStatus("processing");
  };

  const handleStopAudio = () => {
    setAudioStopped(true);
    audioPlayer.stop();
    hasTtsRef.current = false;
    setStatus("idle");
  };

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, currentTranscript, agentResponse]);

  return (
    <main className="px-6 py-10 md:px-12">
      <section className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-semibold md:text-4xl">Sandwich Mode</h1>
        <p className="mt-2 text-sm text-slate-400">STT to agent reasoning to TTS playback.</p>

        <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-5">
          <div ref={scrollRef} className="h-80 space-y-4 overflow-y-auto pr-2">
            {messages.length === 0 ? (
              <p className="pt-24 text-center text-sm text-slate-400">Hold the button to speak.</p>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-xl px-4 py-2 text-sm ${
                      msg.role === "user" ? "bg-orange-500/80" : "bg-white/10"
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))
            )}

            {currentTranscript && (
              <div className="flex justify-end">
                <div className="max-w-[80%] rounded-xl border border-white/20 bg-white/5 px-4 py-2 text-sm text-slate-100">
                  {currentTranscript}
                  <span className="ml-2 align-middle text-[10px] uppercase tracking-widest text-orange-200/70">
                    Transcribing
                  </span>
                </div>
              </div>
            )}

            {agentResponse && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm text-slate-100">
                  {agentResponse}
                  <span className="ml-2 align-middle text-[10px] uppercase tracking-widest text-emerald-200/80">
                    Streaming
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs">
          <span className="text-slate-300">Status</span>
          <div className="flex items-center gap-3">
            <span className="rounded-full bg-white/10 px-3 py-1 text-[10px] uppercase tracking-widest text-slate-200">
              {status}
            </span>
            <button
              onClick={handleStopAudio}
              disabled={status !== "speaking"}
              className="rounded-full bg-white/10 px-3 py-1 text-[10px] uppercase tracking-widest text-slate-200 transition hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Stop Audio
            </button>
          </div>
        </div>

        <div className="mt-8 flex flex-col items-center gap-3">
          <button
            onMouseDown={handleStartRecording}
            onMouseUp={handleStopRecording}
            onTouchStart={handleStartRecording}
            onTouchEnd={handleStopRecording}
            disabled={status === "processing" || status === "speaking"}
            className={`h-24 w-24 rounded-full border border-white/10 text-sm font-semibold transition ${
              isRecording
                ? "bg-orange-500 text-black shadow-lg shadow-orange-500/40"
                : "bg-white/10 hover:bg-white/20"
            } disabled:cursor-not-allowed disabled:opacity-50`}
          >
            {isRecording ? "Recording" : "Hold to Talk"}
          </button>
          <p className="text-xs text-slate-400">Press and hold, then release to send.</p>
        </div>
      </section>
    </main>
  );
}
