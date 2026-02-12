"use client";

import { useCallback, useRef, useState } from "react";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { useAudioPlayer } from "@/hooks/useAudioPlayer";
import { useWebSocket } from "@/hooks/useWebSocket";
import { base64ToArrayBuffer } from "@/lib/audio-utils";
import { getWebSocketUrl } from "@/lib/websocket";

interface ConversationItem {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function RealtimePage() {
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState("");
  const [assistantTranscript, setAssistantTranscript] = useState("");
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [assistantSpeaking, setAssistantSpeaking] = useState(false);

  const userTranscriptBuffer = useRef("");
  const userTranscriptFinalized = useRef(false);
  const assistantBuffer = useRef("");
  const audioPlayer = useAudioPlayer(24000);

  const handleMessage = useCallback(
    (message: any) => {
      switch (message.type) {
        case "speech_started":
          setUserSpeaking(true);
          audioPlayer.stop();
          userTranscriptBuffer.current = "";
          userTranscriptFinalized.current = false;
          setCurrentTranscript("");
          break;
        case "speech_ended":
          setUserSpeaking(false);
          if (userTranscriptBuffer.current && !userTranscriptFinalized.current) {
            setConversation((prev) => [
              ...prev,
              { role: "user", content: userTranscriptBuffer.current, timestamp: new Date() },
            ]);
          }
          userTranscriptBuffer.current = "";
          userTranscriptFinalized.current = false;
          setCurrentTranscript("");
          break;
        case "user_transcript_partial":
          userTranscriptBuffer.current += message.text;
          setCurrentTranscript(userTranscriptBuffer.current);
          break;
        case "user_transcript_final":
          userTranscriptBuffer.current = message.text || userTranscriptBuffer.current;
          if (userTranscriptBuffer.current) {
            setConversation((prev) => [
              ...prev,
              { role: "user", content: userTranscriptBuffer.current, timestamp: new Date() },
            ]);
          }
          userTranscriptFinalized.current = true;
          userTranscriptBuffer.current = "";
          setCurrentTranscript("");
          break;
        case "assistant_text_delta":
          assistantBuffer.current += message.text;
          setAssistantTranscript(assistantBuffer.current);
          break;
        case "assistant_text_done":
          if (message.text) {
            assistantBuffer.current = message.text;
            setAssistantTranscript(assistantBuffer.current);
          }
          break;
        case "audio":
          setAssistantSpeaking(true);
          audioPlayer.playChunk(base64ToArrayBuffer(message.data));
          break;
        case "done":
          setAssistantSpeaking(false);
          if (assistantBuffer.current) {
            setConversation((prev) => [
              ...prev,
              { role: "assistant", content: assistantBuffer.current, timestamp: new Date() },
            ]);
          }
          assistantBuffer.current = "";
          setAssistantTranscript("");
          break;
        case "interrupted":
          setAssistantSpeaking(false);
          audioPlayer.stop();
          break;
        default:
          break;
      }
    },
    [audioPlayer]
  );

  const { connect, disconnect, sendBinary, send } = useWebSocket({
    url: getWebSocketUrl("/ws/realtime"),
    onMessage: handleMessage,
    onClose: () => {
      setIsSessionActive(false);
      setUserSpeaking(false);
      setAssistantSpeaking(false);
    },
  });

  const handleAudioChunk = useCallback(
    (chunk: ArrayBuffer) => {
      sendBinary(chunk);
    },
    [sendBinary]
  );

  const { startRecording, stopRecording } = useAudioRecorder({
    sampleRate: 16000,
    onAudioChunk: handleAudioChunk,
  });

  const startSession = async () => {
    connect();
    await new Promise((resolve) => setTimeout(resolve, 400));
    startRecording();
    setIsSessionActive(true);
  };

  const endSession = () => {
    stopRecording();
    send({ type: "stop" });
    disconnect();
    setIsSessionActive(false);
  };

  return (
    <main className="px-6 py-10 md:px-12">
      <section className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-semibold md:text-4xl">Real-time Mode</h1>
        <p className="mt-2 text-sm text-slate-400">Direct voice-to-voice conversations with GPT-4o Realtime.</p>

        <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="h-80 space-y-4 overflow-y-auto pr-2">
            {conversation.length === 0 && !isSessionActive ? (
              <p className="pt-24 text-center text-sm text-slate-400">Start a session to begin speaking.</p>
            ) : (
              conversation.map((item, idx) => (
                <div
                  key={idx}
                  className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-xl px-4 py-2 text-sm ${
                      item.role === "user" ? "bg-emerald-500/80 text-black" : "bg-white/10"
                    }`}
                  >
                    {item.content}
                  </div>
                </div>
              ))
            )}

            {(userSpeaking || assistantSpeaking || currentTranscript || assistantTranscript) && (
              <div className="rounded-xl border border-white/10 bg-white/5 p-3 text-xs text-slate-200">
                <p className="text-[10px] uppercase tracking-widest text-emerald-200/80">Live</p>
                {(userSpeaking || currentTranscript) && (
                  <p className="mt-1 text-sm text-slate-100">
                    You: {currentTranscript || "Listening..."}
                  </p>
                )}
                {(assistantSpeaking || assistantTranscript) && (
                  <p className="mt-1 text-sm text-slate-100">
                    Assistant: {assistantTranscript || "Speaking..."}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs">
          <span className="text-slate-300">Session</span>
          <span className="rounded-full bg-white/10 px-3 py-1 text-[10px] uppercase tracking-widest text-slate-200">
            {isSessionActive ? "active" : "idle"}
          </span>
        </div>

        <div className="mt-8 flex justify-center">
          {!isSessionActive ? (
            <button
              onClick={startSession}
              className="rounded-full bg-emerald-400 px-8 py-3 text-sm font-semibold text-black transition hover:bg-emerald-300"
            >
              Start Conversation
            </button>
          ) : (
            <button
              onClick={endSession}
              className="rounded-full bg-white/10 px-8 py-3 text-sm font-semibold text-white transition hover:bg-white/20"
            >
              End Conversation
            </button>
          )}
        </div>
      </section>
    </main>
  );
}
