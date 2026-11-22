/**
 * Build and dev-server configuration.
 *
 * `npm run build` writes the app to dist/, which `vitro web` serves; the files use relative URLs, so the
 * app also works from any folder or sub-path. `npm run dev` serves the sources with hot reload and forwards
 * the two routes of the Python server (/api/library and /files/<id>) to a running `vitro web`.
 */

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

/** Where `vitro web --no-browser` listens by default. */
const VITRO_SERVER = 'http://127.0.0.1:8640';

export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    // Fonts must stay files: the server's Content-Security-Policy allows fonts only from the same origin,
    // so a font inlined as a data: URI would be blocked.
    assetsInlineLimit: 0,
  },
  server: {
    port: 5173,
    // changeOrigin rewrites the Host header, which the Python server checks against DNS rebinding.
    proxy: {
      '/api': { target: VITRO_SERVER, changeOrigin: true },
      '/files': { target: VITRO_SERVER, changeOrigin: true },
    },
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
