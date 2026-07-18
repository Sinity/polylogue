import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    ssr: 'src/fixture/ssr.tsx',
    outDir: 'dist/server',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      output: {
        entryFileNames: 'entry.mjs',
      },
    },
  },
});
