import { defineConfig } from 'vite'
import path from 'path'

export default defineConfig({
  // Root is the project root (one level up) where index.html now lives
  root: path.resolve(__dirname, '..'),
  // Public assets stay in sacHacks/public
  publicDir: path.resolve(__dirname, 'public'),
  // Build output goes to sacHacks/dist
  build: {
    outDir: path.resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },
})
