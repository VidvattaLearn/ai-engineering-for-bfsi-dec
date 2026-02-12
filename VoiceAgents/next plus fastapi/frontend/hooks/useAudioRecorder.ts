"use client";

import { useCallback, useRef, useState } from "react";
import { downsampleBuffer, floatTo16BitPCM } from "@/lib/audio-utils";

interface AudioRecorderOptions {
  sampleRate: number;
  onAudioChunk: (chunk: ArrayBuffer) => void;
}

export const useAudioRecorder = ({ sampleRate, onAudioChunk }: AudioRecorderOptions) => {
  const [isRecording, setIsRecording] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    if (isRecording) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(input, audioContext.sampleRate, sampleRate);
      const pcm16 = floatTo16BitPCM(downsampled);
      onAudioChunk(pcm16.buffer);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    audioContextRef.current = audioContext;
    processorRef.current = processor;
    sourceRef.current = source;
    streamRef.current = stream;
    setIsRecording(true);
  }, [isRecording, onAudioChunk, sampleRate]);

  const stopRecording = useCallback(() => {
    if (!isRecording) {
      return;
    }
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    audioContextRef.current?.close();

    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    audioContextRef.current = null;
    setIsRecording(false);
  }, [isRecording]);

  return { isRecording, startRecording, stopRecording };
};
