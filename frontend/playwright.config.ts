import { defineConfig } from '@playwright/test';
export default defineConfig({
 testDir:'./tests',outputDir:'../.local/playwright-results',timeout:60000,workers:1,
 use:{baseURL:'http://127.0.0.1:5173',headless:true,viewport:{width:1440,height:1100},
 launchOptions:{executablePath:'/usr/bin/chromium',args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']},trace:'retain-on-failure'},
 webServer:{command:'bash ../scripts/dev.sh',url:'http://127.0.0.1:5173',reuseExistingServer:true,timeout:30000},
});
