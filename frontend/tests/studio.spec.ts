import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import { generateGroup, photoFile, uploadGroup } from './helpers';

test('starts with a photo and never silently generates a demo person',async({page})=>{
 const jobs:string[]=[];page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/jobs'))jobs.push(r.url())});await page.goto('/');
 await expect(page.getByRole('heading',{name:'Your whole group, a little smaller.'})).toBeVisible();await expect(page.getByRole('button',{name:'Upload your group photo'})).toBeVisible();
 await expect(page.getByRole('button',{name:'Generate whole group',exact:true})).toBeDisabled();await expect(page.getByRole('button',{name:'Download group STL',exact:true})).toBeDisabled();
 for(const label of ['Pose','Outfit','Hat style','Bouquet placement','What are we celebrating?','Head size','X position','Y position','Scale'])await expect(page.getByLabel(label,{exact:true})).toHaveCount(0);
 for(const label of ['Add a little figure','From a photo','Move figure','Rotate figure'])await expect(page.getByRole('button',{name:label,exact:true})).toHaveCount(0);
 await page.waitForTimeout(800);expect(jobs).toEqual([]);
});

test('confirmed photo group generates everyone, preserves photo details, validates and exports real group geometry',async({page},testInfo)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));let submitted:any;page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/jobs'))submitted=r.postDataJSON()});
 await uploadGroup(page);await expect(page.locator('.detection-box')).toHaveCount(2);await expect(page.getByText('Detected people — not crops. The complete group becomes the model.',{exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'Generate whole group',exact:true})).toBeDisabled();await expect(page.getByText('Jacket and trousers · hat',{exact:true})).toBeVisible();await generateGroup(page);
 expect(submitted.figures.map((f:any)=>f.id)).toEqual(['photo-person-1','photo-person-2']);expect(submitted.figures.every((f:any)=>f.included&&f.pose==='photo')).toBe(true);expect(submitted.photo_review).toEqual({detected_people:2,unresolved_people:0,confirmed:true});expect(submitted.figures[0].appearance.accessories).toContain('hat');expect(submitted.figures[1].appearance.accessories).toContain('bouquet');
 const groupDimensions=await page.locator('.dimension-cards').innerText();await page.getByRole('button',{name:'Person 2 Pose from photo'}).click();await expect(page.locator('.dimension-cards')).toHaveText(groupDimensions,{useInnerText:true});await expect(page.getByText('Long dress · bouquet · veil',{exact:true})).toBeVisible();await expect(page.getByRole('button',{name:'Download group STL',exact:true})).toBeEnabled();
 await page.getByRole('button',{name:'Fit scene',exact:true}).click();await page.screenshot({path:testInfo.outputPath('complete-photo-group.png'),fullPage:true});
 await page.getByRole('button',{name:'Run checks',exact:true}).click();await expect(page.getByText('Mesh checks complete for the group.',{exact:true})).toBeVisible();
 const event=page.waitForEvent('download');await page.getByRole('button',{name:'Download group STL',exact:true}).click();const bytes=fs.readFileSync((await(await event).path())!);const triangles=bytes.readUInt32LE(80);expect(triangles).toBeGreaterThan(1000);expect(bytes.length).toBe(84+triangles*50);
 let xmin=Infinity,xmax=-Infinity;for(let i=0;i<triangles;i++)for(let v=0;v<3;v++){const x=bytes.readFloatLE(84+i*50+12+v*12);xmin=Math.min(xmin,x);xmax=Math.max(xmax,x)}expect(xmax-xmin).toBeGreaterThan(70);
 const save=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const project=JSON.parse(fs.readFileSync((await(await save).path())!,'utf8'));expect(project.figures).toHaveLength(2);expect(project.figures.every((f:any)=>f.included)).toBe(true);expect(project.figures[1].landmarks).toHaveLength(33);expect(JSON.stringify(project)).not.toContain(photoFile.name);expect(JSON.stringify(project)).not.toContain('blob:');expect(errors).toEqual([]);
});

test('mobile upload failure stays empty and does not offer manual pose or wardrobe fallbacks',async({page},testInfo)=>{
 await page.setViewportSize({width:390,height:844});await page.goto('/');await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({name:'not-image.png',mimeType:'image/png',buffer:Buffer.from('not image content')});
 await expect(page.getByRole('alert')).toContainText('valid JPG, PNG, or WEBP');await expect(page.getByRole('button',{name:'Generate whole group',exact:true})).toBeDisabled();await expect(page.locator('.subject-list .subject')).toHaveCount(0);await expect(page.getByText('Manual preset',{exact:true})).toHaveCount(0);
 const width=await page.evaluate(()=>({content:document.documentElement.scrollWidth,screen:innerWidth}));expect(width.content).toBeLessThanOrEqual(width.screen);await page.screenshot({path:testInfo.outputPath('photo-first-mobile.png'),fullPage:true});
});
