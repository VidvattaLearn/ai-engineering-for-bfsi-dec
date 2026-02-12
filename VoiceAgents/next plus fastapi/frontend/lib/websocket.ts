export const getWebSocketUrl = (path: string): string => {
  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const url = new URL(path, base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
};
