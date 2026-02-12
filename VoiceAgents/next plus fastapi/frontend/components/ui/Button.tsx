import { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
}

export const Button = ({ variant = "primary", className = "", ...props }: ButtonProps) => {
  const base = "inline-flex items-center justify-center rounded-full px-6 py-3 text-sm font-semibold transition";
  const variants = {
    primary: "bg-emerald-500 text-black hover:bg-emerald-400",
    secondary: "bg-white/10 text-white hover:bg-white/20",
  };
  return <button className={`${base} ${variants[variant]} ${className}`} {...props} />;
};
