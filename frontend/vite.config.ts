import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
const env=(globalThis as typeof globalThis & {process?:{env:Record<string,string|undefined>}}).process?.env || {};
const frontendPort=Number(env.LITTLE_FIGURES_FRONTEND_PORT || 5173);
const backendPort=Number(env.LITTLE_FIGURES_BACKEND_PORT || 8000);
export default defineConfig({
 plugins:[react()],
 build:{rollupOptions:{output:{manualChunks:{graphics:['three'],react:['react','react-dom']}}}},
 server:{host:'0.0.0.0',port:frontendPort,proxy:{'/api':`http://127.0.0.1:${backendPort}`}},
 preview:{port:4173}
});
