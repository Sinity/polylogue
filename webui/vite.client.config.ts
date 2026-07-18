import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist/site',
    emptyOutDir: true,
    cssCodeSplit: false,
    sourcemap: true,
    lib: {
      entry: 'src/fixture/client.tsx',
      formats: ['es'],
      fileName: () => 'assets/webui.js',
    },
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) =>
          assetInfo.name?.endsWith('.css') ? 'assets/webui.css' : 'assets/[name][extname]',
      },
    },
  },
});
