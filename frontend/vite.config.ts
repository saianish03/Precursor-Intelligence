/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // `npm run dev` proxies the API to a local backend; `npm run dev:fixtures` needs none.
  server: { proxy: { "/api": process.env.API_ORIGIN ?? "http://localhost:8000" } },
  test: { globals: true, environment: "jsdom", env: { VITE_FIXTURES: "true" }, setupFiles: ["./src/test/setup.ts"], css: false },
})
