import { defineConfig } from '@playwright/test';
const frontendPort=Number(process.env.LITTLE_FIGURES_FRONTEND_PORT || 5173);
export default defineConfig({
 testDir:'./tests',outputDir:'../.local/playwright-results',timeout:90000,workers:1,
 use:{baseURL:`http://127.0.0.1:${frontendPort}`,headless:true,viewport:{width:1440,height:1100},
 launchOptions:{executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']},trace:'retain-on-failure'},
 webServer:{command:'bash ../scripts/dev.sh',url:`http://127.0.0.1:${frontendPort}`,reuseExistingServer:true,timeout:30000},
});
