interface WaveformVisualizerProps {
  isActive: boolean;
}

export const WaveformVisualizer = ({ isActive }: WaveformVisualizerProps) => {
  return (
    <div className="flex h-10 items-center gap-2">
      {Array.from({ length: 12 }).map((_, index) => (
        <span
          key={index}
          className={`w-1 rounded-full bg-emerald-400/70 ${
            isActive ? "animate-pulse" : "opacity-40"
          }`}
          style={{ height: `${8 + (index % 4) * 6}px` }}
        />
      ))}
    </div>
  );
};
