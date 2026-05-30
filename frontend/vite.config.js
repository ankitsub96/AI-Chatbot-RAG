import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/rag': 'http://localhost:8000',
      '/extract': 'http://localhost:8000',
    }
  }
})
