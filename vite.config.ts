import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '127.0.0.1',
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  build: {
    // Monaco is cached separately from the application and language resources.
    chunkSizeWarningLimit: 3000,
    rolldownOptions: { output: { codeSplitting: { groups: [{ name: 'editor', test: /node_modules\/monaco-editor\// }] } } },
  },
})
