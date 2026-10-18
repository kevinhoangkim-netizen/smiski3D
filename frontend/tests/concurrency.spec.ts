import { test, expect } from '@playwright/test';
import { mesh, photoFile, photoResponse, projectFile, uploadGroup } from './helpers';
const deferred=()=>{let resolve!:()=>void;const promise=new Promise<void>(r=>{resolve=r});return {promise,resolve}};

for(const action of ['new-photo','remove-photo','import'] as const)test(`late person analysis cannot replace a subsequent ${action}`,async({page})=>{
 const started=deferred(),release=deferred(),handled=deferred();let calls=0;
 await page.route('**/api/analyze',async route=>{calls++;if(calls>1)return route.fulfill({json:photoResponse(2)});started.resolve();await release.promise;try{await route.fulfill({json:photoResponse(1,{warnings:['Stale response must not appear.']})})}catch{/* Explicit cancellation aborts this request. */}finally{handled.resolve()}});
 await page.goto('/');await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(photoFile);await started.promise;
 if(action==='new-photo'){await page.getByRole('button',{name:'New photo',exact:true}).click();await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({...photoFile,name:'replacement-photo.png'});await expect(page.getByText('2 people found',{exact:true})).toBeVisible()}
 else if(action==='remove-photo')await page.getByRole('button',{name:'Remove reference photo',exact:true}).click();
 else{await page.locator('input[type=file][accept=".json,application/json"]').setInputFiles(projectFile('Keep this legacy scene',115));await expect(page.getByRole('button',{name:'Keep this legacy scene Saved legacy pose'})).toBeVisible()}
 release.resolve();await handled.promise;await page.waitForTimeout(250);await expect(page.getByText('Stale response must not appear.',{exact:true})).toHaveCount(0);
 if(action==='new-photo'){await expect(page.getByText('2 people found',{exact:true})).toBeVisible();await expect(page.locator('.subject-list .subject')).toHaveCount(2)}else if(action==='remove-photo'){await expect(page.locator('.subject-list .subject')).toHaveCount(0);await expect(page.getByRole('button',{name:'Generate whole group',exact:true})).toBeDisabled()}else{await expect(page.getByRole('button',{name:'Keep this legacy scene Saved legacy pose'})).toBeVisible();await expect(page.getByLabel('Tallest figure height',{exact:true})).toHaveValue('115')}
});

test('an older generation cannot overwrite a replacement group',async({page})=>{
 const started=deferred(),release=deferred(),handled=deferred();let first=true;
 await uploadGroup(page);await page.route('**/api/jobs',async route=>{if(!first)return route.continue();first=false;return route.fulfill({json:{id:'delayed-generation'}})});await page.route('**/api/jobs/delayed-generation',async route=>{started.resolve();await release.promise;try{await route.fulfill({json:{state:'ready',stage:'Old scene ready',result:{figures:[mesh('photo-person-1','Old person 1'),mesh('photo-person-2','Old person 2')]}}})}catch{/* Superseded polls are aborted. */}finally{handled.resolve()}});
 await page.getByRole('checkbox',{name:/Everyone in the photo is included/}).check();await page.getByRole('button',{name:'Generate whole group',exact:true}).click();await started.promise;
 await page.locator('input[type=file][accept=".json,application/json"]').setInputFiles(projectFile('Keep the replacement',130));await expect(page.getByRole('button',{name:'Download group STL',exact:true})).toBeEnabled({timeout:60000});release.resolve();await handled.promise;await page.waitForTimeout(250);
 await expect(page.getByRole('button',{name:'Keep the replacement Saved legacy pose'})).toBeVisible();await expect(page.locator('.dimension-cards').getByText('130.0',{exact:true})).toBeVisible();await expect(page.getByRole('button',{name:'Download group STL',exact:true})).toBeEnabled();
});

test('a generation missing a detected person cannot become an exportable group',async({page})=>{
 await uploadGroup(page);await page.route('**/api/jobs',route=>route.fulfill({json:{id:'incomplete-group'}}));await page.route('**/api/jobs/incomplete-group',route=>route.fulfill({json:{state:'ready',stage:'Ready',result:{figures:[mesh('photo-person-1')]}}}));await page.getByRole('checkbox',{name:/Everyone in the photo is included/}).check();await page.getByRole('button',{name:'Generate whole group',exact:true}).click();await expect(page.getByRole('alert')).toContainText('missing a person');await expect(page.getByRole('button',{name:'Download group STL',exact:true})).toBeDisabled();await expect(page.getByRole('button',{name:'Shareable 3D preview',exact:true})).toBeDisabled();
});
