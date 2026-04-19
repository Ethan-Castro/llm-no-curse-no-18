import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          purple: "#7b2ff7",
          blue: "#1565c0",
          soft: "#f8f9fc",
          ink: "#1a1a2e",
        },
      },
      fontFamily: {
        sans: [
          "var(--font-inter)",
          "Inter",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "sans-serif",
        ],
      },
      animation: {
        blob: "blobFloat 9s ease-in-out infinite",
        "bounce-slow": "bounce 1.4s infinite",
      },
      keyframes: {
        blobFloat: {
          "0%, 100%": { transform: "translate(0, 0) scale(1)" },
          "25%": { transform: "translate(20px, -15px) scale(1.1)" },
          "50%": { transform: "translate(-10px, 20px) scale(0.95)" },
          "75%": { transform: "translate(15px, 10px) scale(1.05)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
