import preact from '@preact/preset-vite';
import { defineConfig } from 'vitest/config';
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
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
});
