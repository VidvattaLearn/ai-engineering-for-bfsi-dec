"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface WebSocketOptions {
  url: string;
  onMessage?: (data: any) => void;
  onClose?: () => void;
}

export const useWebSocket = ({ url, onMessage, onClose }: WebSocketOptions) => {
  const socketRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = useCallback(() => {
    if (socketRef.current) {
      return;
    }
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";

    socket.addEventListener("open", () => setIsConnected(true));
    socket.addEventListener("close", () => {
      setIsConnected(false);
      socketRef.current = null;
      onClose?.();
    });
    socket.addEventListener("message", (event) => {
      if (!onMessage) {
        return;
      }
      if (typeof event.data === "string") {
        try {
          onMessage(JSON.parse(event.data));
        } catch {
          onMessage(event.data);
        }
      } else {
        onMessage(event.data);
      }
    });

    socketRef.current = socket;
  }, [url, onMessage, onClose]);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setIsConnected(false);
  }, []);

  const send = useCallback((payload: any) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(payload));
    }
  }, []);

  const sendBinary = useCallback((payload: ArrayBuffer) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(payload);
    }
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    isConnected,
    connect,
    disconnect,
    send,
    sendBinary,
  };
};
