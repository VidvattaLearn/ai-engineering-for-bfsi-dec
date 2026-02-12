import { ReactNode } from "react";

interface CardProps {
  title: string;
  children: ReactNode;
}

export const Card = ({ title, children }: CardProps) => {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
      <h3 className="text-sm uppercase tracking-[0.2em] text-slate-400">{title}</h3>
      <div className="mt-4 text-sm text-slate-200">{children}</div>
    </div>
  );
};
