import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import { generateGroup, uploadGroup } from './helpers';
import { createPreviewHTML, safePreviewJSON, type PreviewScene } from '../src/sharing';

const scene:PreviewScene={
 meshes:[{name:'A <special> keepsake',positions:[-20,-10,0,20,-10,0,20,10,0,-20,10,0,-20,-10,40,20,-10,40,20,10,40,-20,10,40],normals:[-1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,-1,1,1],indices:[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7],color:'#d5e69d'}],
 dimensions:{width:40,depth:20,height:40},center:[0,0,20],radius:30,
};

test('downloadable preview works offline with orbit, zoom, image export and no network',async({page},testInfo)=>{
 const path=testInfo.outputPath('offline-preview.html');fs.writeFileSync(path,createPreviewHTML(scene));
 const errors:string[]=[],network:string[]=[];page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>{if(/^https?:/.test(request.url()))network.push(request.url())});
 // The managed test browser blocks file:// navigation. Load the exact saved file
 // into an offline page to verify the dependency-free viewer without bypassing that policy.
 await page.context().setOffline(true);await page.setContent(fs.readFileSync(path,'utf8'));
 await expect(page.getByText('40.0 × 20.0 × 40.0 mm (width × depth × height)')).toBeVisible();
 await expect(page.getByText('Private offline preview · geometry only · no source photo')).toBeVisible();
 await page.getByRole('button',{name:'Pause rotation'}).click();
 await page.waitForTimeout(100);
 const before=await page.locator('canvas').evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL());
 const pixels=await page.locator('canvas').evaluate((canvas:HTMLCanvasElement)=>{const gl=canvas.getContext('webgl')!;const p=new Uint8Array(4);gl.readPixels(Math.floor(canvas.width/2),Math.floor(canvas.height/2),1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);return Array.from(p)});
 expect(pixels[0]).toBeLessThan(230); // Filled model, rather than just the pale background.
 await page.mouse.move(720,550);await page.mouse.down();await page.mouse.move(960,610,{steps:8});await page.mouse.up();await page.waitForTimeout(100);
 const after=await page.locator('canvas').evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL());expect(after).not.toBe(before);
 await page.locator('canvas').focus();await page.keyboard.press('+');await page.waitForTimeout(100);
 const zoomed=await page.locator('canvas').evaluate((canvas:HTMLCanvasElement)=>canvas.toDataURL());expect(zoomed).not.toBe(after);
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save image'}).click();const image=await download;expect(image.suggestedFilename()).toBe('little-figures.png');expect(fs.statSync((await image.path())!).size).toBeGreaterThan(1000);
 expect(errors).toEqual([]);expect(network).toEqual([]);
});

test('scene metadata cannot inject scripts into the downloadable HTML',async({page},testInfo)=>{
 const attack='</script><script>window.__injected=true</script><img src="https://example.com/leak">';
 expect(safePreviewJSON({name:attack})).not.toContain('<');
 const path=testInfo.outputPath('escaped-preview.html');fs.writeFileSync(path,createPreviewHTML({...scene,meshes:[{...scene.meshes[0],name:attack}]}));
 const requests:string[]=[];page.on('request',request=>{if(/^https?:/.test(request.url()))requests.push(request.url())});
 await page.context().setOffline(true);await page.setContent(fs.readFileSync(path,'utf8'));await expect(page.locator('canvas')).toBeVisible();
 expect(await page.evaluate(()=>('__injected' in window))).toBe(false);expect(requests).toEqual([]);
 expect(await page.locator('#scene').evaluate(el=>JSON.parse(el.textContent!).meshes[0].name)).toBe(attack);
});

test('photo group exports real PNG, GLB, private HTML and a turntable video',async({page})=>{
 test.setTimeout(120000);const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));await uploadGroup(page);await generateGroup(page);
 async function exported(label:string){const event=page.waitForEvent('download');await page.getByRole('button',{name:label,exact:true}).click();const download=await event;return fs.readFileSync((await download.path())!)}
 const image=await exported('Save image');expect(image.subarray(0,8).toString('hex')).toBe('89504e470d0a1a0a');
 const html=(await exported('Shareable 3D preview')).toString('utf8');expect(html).not.toContain('src="http');expect(html).not.toContain('data:image/');expect(html).not.toContain('private-group-reference.png');
 const payload=JSON.parse(html.match(/<script id="scene" type="application\/json">([\s\S]*?)<\/script>/)![1]);expect(payload.meshes.some((m:{name:string})=>m.name==='Person 1')).toBe(true);expect(payload.meshes.some((m:{name:string})=>m.name==='Person 2')).toBe(true);expect(payload.dimensions.width).toBeGreaterThan(70);expect(payload.meshes.filter((m:{color:string})=>m.color==='#35422a')).toHaveLength(2);
 const glb=await exported('Download color preview (.glb)');expect(glb.toString('ascii',0,4)).toBe('glTF');const json=JSON.parse(glb.toString('utf8',20,20+glb.readUInt32LE(12)));const root=json.nodes[json.scenes[json.scene??0].nodes[0]],unitScale=root.scale?.[0]??Math.hypot(...root.matrix.slice(0,3));expect(unitScale).toBeCloseTo(.001,8);expect(json.nodes.some((node:{name?:string})=>node.name==='Person 1')).toBe(true);expect(json.nodes.some((node:{name?:string})=>node.name==='Person 2')).toBe(true);
 const supportsVideo=await page.evaluate(()=>typeof MediaRecorder!=='undefined'&&MediaRecorder.isTypeSupported('video/webm'));if(supportsVideo){const video=await exported('Turntable video');expect(video.subarray(0,4).toString('hex')).toBe('1a45dfa3');expect(video.length).toBeGreaterThan(10000)}expect(errors).toEqual([]);
});

test('shared pedestal and person inspection retain everyone in the private preview',async({page},testInfo)=>{
 await uploadGroup(page);await page.getByLabel('Group pedestal',{exact:true}).selectOption('rectangle');await page.getByLabel('Names, a date, a little message').fill('Together');await generateGroup(page);await page.getByRole('button',{name:'Person 2 Pose from photo'}).click();await page.getByRole('button',{name:'Fit scene',exact:true}).click();await page.screenshot({path:testInfo.outputPath('photo-group-shared-pedestal.png'),fullPage:true});
 const event=page.waitForEvent('download');await page.getByRole('button',{name:'Shareable 3D preview',exact:true}).click();const html=fs.readFileSync((await(await event).path())!,'utf8');const data=JSON.parse(html.match(/<script id="scene" type="application\/json">([\s\S]*?)<\/script>/)![1]);expect(data.meshes.filter((m:{name:string})=>m.name==='Shared pedestal')).toHaveLength(1);expect(data.meshes.some((m:{name:string})=>m.name==='Person 1')).toBe(true);expect(data.meshes.some((m:{name:string})=>m.name==='Person 2')).toBe(true);expect(data.dimensions.width).toBeGreaterThan(70);const offline=await page.context().newPage();await offline.setContent(html);await expect(offline.locator('#error')).toBeHidden();await expect(offline.getByRole('button',{name:'Pause rotation'})).toBeVisible();await offline.close();
});
