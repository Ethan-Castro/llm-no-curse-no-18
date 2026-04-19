import Chat from "@/components/Chat";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-8 md:py-12">
      {/* Header */}
      <header className="mb-10 flex flex-col items-center text-center">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-sky-200/70 bg-white/60 px-3 py-1 backdrop-blur">
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-sky-500" />
          </span>
          <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-600">
            Live demo
          </span>
        </div>

        <h1 className="bg-gradient-to-b from-slate-900 to-sky-700 bg-clip-text text-6xl font-bold leading-[0.95] tracking-tight text-transparent md:text-7xl">
          KidsChat
        </h1>

        <p className="mt-5 max-w-xl text-base leading-relaxed text-slate-600 md:text-lg">
          A 180M-parameter language model trained on a corpus with{" "}
          <span className="font-semibold text-slate-800">zero profanity</span>{" "}
          and{" "}
          <span className="font-semibold text-slate-800">zero adult content</span>
          .
        </p>

        <div className="mt-6 flex items-center gap-5 text-xs font-medium text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="font-mono text-sky-600">180M</span> params
          </span>
          <span className="h-3 w-px bg-slate-300" />
          <span className="flex items-center gap-1.5">
            <span className="font-mono text-sky-600">12</span> layers
          </span>
          <span className="h-3 w-px bg-slate-300" />
          <span className="flex items-center gap-1.5">
            <span className="font-mono text-sky-600">32k</span> vocab
          </span>
        </div>
      </header>

      {/* Chat */}
      <Chat />

      {/* Footer */}
      <footer className="mt-8 text-center text-xs text-slate-500">
        <p>Powered by nanochat · Hosted on Modal · Streaming via SSE</p>
        <p className="mt-1">Cold start ~10-15s • Scales to zero when idle</p>
      </footer>
    </main>
  );
}
