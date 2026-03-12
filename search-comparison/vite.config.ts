import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/api-a': {
        target: 'http://localhost:8301',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api-a/, ''),
      },
      '/api-b': {
        target: 'http://localhost:8300',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api-b/, ''),
      },
    },
  },
})
