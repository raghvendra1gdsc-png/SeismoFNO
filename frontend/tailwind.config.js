/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          deep: "#FFFFFF",
          base: "#F4F4F4",
          surface: "#FFFFFF",
          elevated: "#FFFFFF",
        },
        panel: {
          base: "#FFFFFF",
          hover: "#F4F4F4",
          active: "#EDF5FF",
          bg: "#FFFFFF",
        },
        border: {
          subtle: "#E0E0E0",
          medium: "#C6C6C6",
          strong: "#8D8D8D",
        },
        text: {
          primary: "#161616",
          secondary: "#525252",
          muted: "#8D8D8D",
          disabled: "#A8A8A8",
        },
        fno: {
          DEFAULT: "#0F62FE", // Carbon blue-60
          light: "#4589FF",
          dark: "#0043CE",
          dim: "#EDF5FF", // Carbon blue-10
        },
        carbon: {
          blue60: "#0F62FE",
          blue10: "#EDF5FF",
          blue70: "#0353E9",
          green60: "#198038",
          amber60: "#B28600",
          red60: "#DA1E28",
          purple60: "#8A3FFA",
          gray10: "#F4F4F4",
          gray20: "#E0E0E0",
          gray50: "#8D8D8D",
          gray70: "#525252",
          gray100: "#161616",
        },
        opensees: {
          DEFAULT: "#B28600", // Carbon amber-60
          light: "#D2A106",
          dark: "#8E6A00",
          dim: "#FFF8E1",
        },
        energy: {
          DEFAULT: "#198038", // Carbon green-60
          light: "#24A148",
          dark: "#0E6027",
          dim: "#DEFBE6",
        },
        danger: {
          DEFAULT: "#DA1E28", // Carbon red-60
          light: "#FA4D56",
          dark: "#A2191F",
          dim: "#FFD7D9",
        }
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["IBM Plex Mono", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
}
