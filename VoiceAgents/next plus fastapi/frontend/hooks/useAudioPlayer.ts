"use client";

import { useCallback, useRef } from "react";

export const useAudioPlayer = (sampleRate: number, onIdle?: () => void) => {
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextTimeRef = useRef(0);
  const sourcesRef = useRef<AudioBufferSourceNode[]>([]);

  const ensureContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate });
      nextTimeRef.current = audioContextRef.current.currentTime;
    }
    return audioContextRef.current;
  };

  const playChunk = useCallback((chunk: ArrayBuffer) => {
    const context = ensureContext();
    if (context.state === "suspended") {
      context.resume();
    }
    const pcm16 = new Int16Array(chunk);
    const audioBuffer = context.createBuffer(1, pcm16.length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);

    for (let i = 0; i < pcm16.length; i += 1) {
      channelData[i] = pcm16[i] / 0x8000;
    }

    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.onended = () => {
      sourcesRef.current = sourcesRef.current.filter((node) => node !== source);
      if (sourcesRef.current.length === 0) {
        onIdle?.();
      }
    };
    sourcesRef.current = [...sourcesRef.current, source];

    const startTime = Math.max(context.currentTime, nextTimeRef.current);
    source.start(startTime);
    nextTimeRef.current = startTime + audioBuffer.duration;
  }, [sampleRate]);

  const stop = useCallback(() => {
    sourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch {
        // Ignore nodes that already stopped.
      }
    });
    sourcesRef.current = [];
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    nextTimeRef.current = 0;
    onIdle?.();
  }, []);

  return { playChunk, stop };
};
