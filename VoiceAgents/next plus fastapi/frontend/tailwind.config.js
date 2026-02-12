/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Space Grotesk'", "sans-serif"],
      },
      colors: {
        ink: "#0e0f13",
        dusk: "#1f2937",
        ember: "#f97316",
        mint: "#10b981",
      },
    },
  },
  plugins: [],
};
