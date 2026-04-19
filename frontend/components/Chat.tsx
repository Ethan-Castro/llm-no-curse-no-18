"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Trash2, Loader2, AlertCircle } from "lucide-react";

type Role = "user" | "assistant";
type Message = { role: Role; content: string };

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, streaming]);

  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, []);

  useEffect(resizeTextarea, [input, resizeTextarea]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;

    setError(null);
    setInput("");

    const nextMessages: Message[] = [
      ...messages,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ];
    setMessages(nextMessages);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_URL}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: nextMessages
            .slice(0, -1)
            .map((m) => ({ role: m.role, content: m.content })),
          stream: true,
          temperature: 0.8,
          top_k: 50,
          max_tokens: 512,
        }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(
          `Server error ${res.status}. Is the backend running?`,
        );
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.slice(5).trim();
          if (data === "[DONE]") continue;

          try {
            const chunk = JSON.parse(data);
            // Nanochat returns {token: "...", gpu: N} or {done: true}
            const token: string = chunk?.token ?? "";
            if (!token) continue;

            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last?.role === "assistant") {
                copy[copy.length - 1] = {
                  ...last,
                  content: last.content + token,
                };
              }
              return copy;
            });
          } catch {
            // ignore malformed frames
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError(
          (err as Error).message ||
            "Could not reach the chat server. Check your connection.",
        );
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [input, messages, streaming]);

  const stop = () => {
    abortRef.current?.abort();
  };

  const clear = () => {
    if (streaming) return;
    setMessages([]);
    setError(null);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden rounded-3xl border border-sky-200 bg-gradient-to-br from-sky-50 via-blue-50 to-cyan-50 shadow-xl">
      {/* Messages area */}
      <div
        ref={scrollerRef}
        className="scrollbar-thin flex-1 space-y-4 overflow-y-auto px-4 py-6 md:px-8 md:py-8"
        style={{ minHeight: "50vh", maxHeight: "65vh" }}
      >
        {messages.length === 0 && (
          <div className="mx-auto flex max-w-md flex-col items-center justify-center py-16 text-center">
            <div className="mb-4 rounded-full bg-gradient-to-br from-sky-100 to-cyan-100 p-4">
              <svg
                className="h-8 w-8 text-sky-600"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z" />
              </svg>
            </div>
            <p className="text-xl font-semibold text-slate-800">
              Welcome to KidsChat
            </p>
            <p className="mt-2 text-sm text-slate-600">
              Ask me anything! I'm trained to be helpful and safe.
            </p>
            <div className="mt-4 space-y-2 text-left">
              <p className="text-xs font-medium text-slate-500">Try asking:</p>
              <ul className="text-xs text-slate-600">
                <li>• "What's a fun fact about space?"</li>
                <li>• "How do I make a paper airplane?"</li>
                <li>• "Tell me a short story"</li>
              </ul>
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <MessageBubble
            key={i}
            message={m}
            isStreaming={
              streaming &&
              i === messages.length - 1 &&
              m.role === "assistant"
            }
          />
        ))}
      </div>

      {/* Error banner */}
      {error && (
        <div className="border-t border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Connection error</p>
            <p className="mt-0.5 text-xs text-red-600">{error}</p>
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-sky-200 bg-white/60 backdrop-blur-sm p-4 md:p-5">
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask KidsChat something..."
            rows={1}
            disabled={streaming}
            className="flex-1 resize-none rounded-2xl border border-sky-200 bg-white px-4 py-3 text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-sky-400 focus:ring-2 focus:ring-sky-300/50 disabled:opacity-50 disabled:bg-slate-50"
          />
          {streaming ? (
            <button
              onClick={stop}
              className="flex items-center gap-2 rounded-2xl bg-gradient-to-br from-orange-500 to-red-500 px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:shadow-xl hover:scale-105 active:scale-95"
            >
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              onClick={send}
              disabled={!input.trim()}
              className="flex items-center gap-2 rounded-2xl bg-gradient-to-br from-sky-500 to-cyan-500 px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:shadow-xl hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-lg disabled:hover:scale-100"
            >
              <Send className="h-4 w-4" />
              <span>Send</span>
            </button>
          )}
        </div>
        <div className="mt-3 flex items-center justify-between">
          <span className="text-xs text-slate-500">
            Shift+Enter for new line
          </span>
          <button
            onClick={clear}
            disabled={streaming || messages.length === 0}
            className="flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs text-slate-600 transition hover:bg-sky-100 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  isStreaming,
}: {
  message: Message;
  isStreaming: boolean;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-3xl px-5 py-3.5 text-sm leading-relaxed shadow-md transition-all ${
          isUser
            ? "rounded-br-sm bg-gradient-to-br from-sky-500 to-cyan-500 text-white font-medium"
            : "rounded-bl-sm border border-sky-200 bg-white text-slate-800"
        }`}
      >
        {message.content || isStreaming ? (
          <span className="whitespace-pre-wrap break-words">
            {message.content}
            {isStreaming && (
              <span className="ml-1 inline-block h-2 w-2 rounded-full bg-current opacity-60 animate-pulse" />
            )}
          </span>
        ) : null}
      </div>
    </div>
  );
}
