import { test, expect } from '@playwright/test';
import fs from 'node:fs';

test('manual studio generates, edits, validates, saves and exports real geometry',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByRole('heading',{name:'Make a moment tangible.'})).toBeVisible();
 const groupExport=page.getByRole('button',{name:'Download STL',exact:true});
 await expect(groupExport).toBeEnabled({timeout:30000});
 await expect(page.locator('canvas')).toBeVisible();
 await expect(page.getByText('WebGL', {exact:false})).toHaveCount(0);
 await page.waitForTimeout(1000);
 await page.screenshot({path:'../examples/studio-desktop.png',fullPage:true});
 const height=page.getByRole('spinbutton',{name:'Target overall height'});
 await height.fill('100');
 await expect(groupExport).toBeDisabled();
 await page.getByRole('button',{name:'Generate figures',exact:true}).click();
 await expect(groupExport).toBeEnabled();
 await expect(page.locator('.dimension-cards').getByText('100.0')).toBeVisible();
 await page.getByRole('button',{name:'Run checks'}).click();
 await expect(page.getByText('Mesh checks complete. Review the results below.')).toBeVisible();
 const downloadEvent=page.waitForEvent('download');await groupExport.click();const download=await downloadEvent;
 const path=await download.path();expect(path).toBeTruthy();const bytes=fs.readFileSync(path!);
 const count=bytes.readUInt32LE(80);expect(count).toBeGreaterThan(1000);expect(bytes.length).toBe(84+count*50);
 let min=Infinity,max=-Infinity;for(let i=0;i<count;i++)for(let v=0;v<3;v++){const z=bytes.readFloatLE(84+i*50+12+v*12+8);min=Math.min(min,z);max=Math.max(max,z)}
 expect(max-min).toBeCloseTo(100,2);
 await page.getByRole('button',{name:'Add a little figure'}).click();
 await expect(page.getByRole('checkbox',{name:'Include Little figure 2'})).toBeChecked();
 await page.getByRole('checkbox',{name:'Include Little figure 2'}).uncheck();
 await page.getByRole('button',{name:'Generate figures',exact:true}).click();
 await expect(groupExport).toBeEnabled();
 const saveEvent=page.waitForEvent('download');await page.getByRole('button',{name:'Save project'}).click();const saved=await saveEvent;
 const project=JSON.parse(fs.readFileSync((await saved.path())!,'utf8'));expect(project.version).toBe(1);expect(project.figures).toHaveLength(2);expect(project.figures[1].included).toBe(false);
 expect(errors).toEqual([]);
});

test('mobile layout and explicit photo fallback',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('/');
 await expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeEnabled({timeout:30000});
 await page.getByRole('button',{name:'From a photo'}).click();
 await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({name:'not-image.png',mimeType:'image/png',buffer:Buffer.from('not image content')});
 await expect(page.getByRole('alert')).toContainText('valid JPG, PNG, or WEBP');
 await expect(page.getByText('Manual preset', {exact:true})).toBeVisible();
 const width=await page.evaluate(()=>({content:document.documentElement.scrollWidth,screen:innerWidth}));expect(width.content).toBeLessThanOrEqual(width.screen);
 await page.screenshot({path:'../examples/studio-mobile.png',fullPage:true});
});

test('group layout, transforms, undo, project reopen and GLB export',async({page})=>{
 await page.goto('/');await expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeEnabled({timeout:30000});
 await page.getByRole('spinbutton',{name:'Scale',exact:true}).fill('150');
 await expect(page.locator('.dimension-cards').getByText('135.0')).toBeVisible();
 await page.getByRole('button',{name:'Undo',exact:true}).click();
 await expect(page.locator('.dimension-cards').getByText('90.0')).toBeVisible();
 await page.getByRole('button',{name:'Add a little figure'}).click();
 await page.getByRole('button',{name:'Generate figures',exact:true}).click();
 const combined=page.getByRole('button',{name:'Download group STL'});await expect(combined).toBeEnabled();
 const groupEvent=page.waitForEvent('download');await combined.click();const group=await groupEvent;const binary=fs.readFileSync((await group.path())!);expect(binary.readUInt32LE(80)).toBeGreaterThan(10000);
 const glbEvent=page.waitForEvent('download');await page.getByRole('button',{name:'Download color preview (.glb)'}).click();const glb=fs.readFileSync((await (await glbEvent).path())!);expect(glb.toString('ascii',0,4)).toBe('glTF');
 const glbJson=JSON.parse(glb.toString('utf8',20,20+glb.readUInt32LE(12)));const root=glbJson.nodes[glbJson.scenes[glbJson.scene??0].nodes[0]];const unitScale=root.scale?.[0]??Math.hypot(...root.matrix.slice(0,3));expect(unitScale).toBeCloseTo(.001,8);
 const saveEvent=page.waitForEvent('download');await page.getByRole('button',{name:'Save project'}).click();const save=await saveEvent;const savedPath=(await save.path())!;
 await page.getByRole('button',{name:'Restore all defaults',exact:true}).click();await expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeEnabled();
 await page.locator('input[type=file][accept=".json,application/json"]').setInputFiles(savedPath);
 await expect(combined).toBeEnabled();await expect(page.getByRole('checkbox',{name:'Include Little figure 2'})).toBeChecked();
});

test('real local photo detects two subjects and exports only included people',async({page,request})=>{
 test.skip(!fs.existsSync('../.local/two-person-test.jpg'),'Optional public reference fixture has not been downloaded.');
 const health=await (await request.get('/api/health')).json();test.skip(!health.pose_available,'Official pose model is unavailable; manual workflow remains tested.');
 await page.goto('/');await expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeEnabled({timeout:30000});
 await page.getByRole('button',{name:'From a photo'}).click();
 await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles('../.local/two-person-test.jpg');
 await expect(page.getByRole('checkbox',{name:'Include Person 1'})).toBeVisible({timeout:30000});
 await expect(page.getByRole('checkbox',{name:'Include Person 2'})).toBeChecked();
 await expect(page.getByText('Estimated photo pose',{exact:true})).toHaveCount(2);
 await page.getByRole('checkbox',{name:'Include Person 2'}).uncheck();
 await page.getByRole('button',{name:'Generate figures',exact:true}).click();
 const single=page.getByRole('button',{name:'Download STL',exact:true});await expect(single).toBeEnabled({timeout:30000});
 const exported=page.waitForEvent('download');await single.click();const bytes=fs.readFileSync((await (await exported).path())!);expect(bytes.readUInt32LE(80)).toBeGreaterThan(10000);
 await expect(page.getByText('Pose interpretation',{exact:true})).toBeVisible();
});
