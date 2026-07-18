import preact from '@preact/preset-vite';
import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

const daemonUrl = process.env.POLYLOGUE_DAEMON_URL ?? 'http://127.0.0.1:8787';

export default defineConfig({
  plugins: [preact()],
  base: '/app/assets/',
  build: {
    outDir: '../polylogue/daemon/static/dist',
    emptyOutDir: true,
    manifest: 'manifest.json',
    sourcemap: false,
    rollupOptions: {
      input: {
        'archive-overview': fileURLToPath(
          new URL('./src/entrypoints/archive-overview.tsx', import.meta.url),
        ),
        observability: fileURLToPath(new URL('./src/entrypoints/observability.tsx', import.meta.url)),
        'session-list': fileURLToPath(new URL('./src/entrypoints/session-list.tsx', import.meta.url)),
        'session-read': fileURLToPath(new URL('./src/entrypoints/session-read.tsx', import.meta.url)),
      },
      output: {
        entryFileNames: '[name]-[hash].js',
        chunkFileNames: '[name]-[hash].js',
        assetFileNames: '[name]-[hash][extname]',
      },
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: daemonUrl,
        changeOrigin: false,
      },
    },
  },
});

// Vitest configuration lives in ./vitest.config.ts (single source of truth for
// the test include/setup/environment so it is shared by every src/**/*.test.*
// file regardless of which feature added it).
