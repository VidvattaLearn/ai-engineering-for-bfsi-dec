import Link from "next/link";

const modes = [
  {
    title: "Sandwich Mode",
    description: "Streamed STT, agent reasoning, and TTS playback.",
    href: "/sandwich",
    accent: "from-orange-500/20 to-orange-500/5",
  },
  {
    title: "Real-time Mode",
    description: "Direct voice-to-voice conversation with GPT-4o Realtime.",
    href: "/realtime",
    accent: "from-emerald-500/20 to-emerald-500/5",
  },
];

export default function HomePage() {
  return (
    <main className="px-6 py-12 md:px-12">
      <section className="mx-auto max-w-4xl">
        <p className="text-xs uppercase tracking-[0.3em] text-orange-200/80">Voice Agents Lab</p>
        <h1 className="mt-4 text-4xl font-semibold md:text-6xl">
          Design, test, and compare voice-first AI architectures.
        </h1>
        <p className="mt-4 max-w-2xl text-base text-slate-300 md:text-lg">
          Choose a mode to explore streaming pipelines or end-to-end voice sessions powered by Azure OpenAI and ElevenLabs.
        </p>
      </section>

      <section className="mx-auto mt-12 grid max-w-4xl gap-6 md:grid-cols-2">
        {modes.map((mode) => (
          <Link
            key={mode.title}
            href={mode.href}
            className={`group rounded-2xl border border-white/10 bg-gradient-to-br ${mode.accent} p-6 transition hover:-translate-y-1 hover:border-white/20`}
          >
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">{mode.title}</h2>
              <span className="text-sm text-slate-300 group-hover:text-white">Open</span>
            </div>
            <p className="mt-3 text-sm text-slate-300">{mode.description}</p>
            <div className="mt-6 text-xs text-slate-400">Launch experience</div>
          </Link>
        ))}
      </section>
    </main>
  );
}
