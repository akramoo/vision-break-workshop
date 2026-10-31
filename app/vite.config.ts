import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' keeps every URL relative so the same dist/ works on GitHub Pages,
// `npx serve dist`, the bundled serve-offline.mjs and a USB stick.
// public/ (models, data, serve-offline.mjs) is copied into dist/ automatically.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 2500,
  },
  server: { port: 5173 },
  preview: { port: 4173, strictPort: true },
});
