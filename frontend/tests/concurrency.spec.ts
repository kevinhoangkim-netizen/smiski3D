import { test, expect, type Page } from '@playwright/test';

const ready=(page:Page)=>expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeEnabled({timeout:45000});
const deferred=()=>{let resolve!:()=>void;const promise=new Promise<void>(r=>{resolve=r;});return {promise,resolve};};
const photoFile={name:'delayed-photo.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j7X0AAAAASUVORK5CYII=','base64')};
const referenceResponse={subjects:[{id:'stale-analysis-person',name:'Stale subject',landmarks:Array.from({length:33},()=>({x:.5,y:.5,z:0,visibility:1})),bbox:[.1,.1,.9,.9],confidence:.9,warnings:['Stale analysis warning'],photo_aspect:1.5}],warnings:['Stale general warning']};
function projectFile(name:string,height:number){return {name:'replacement.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({version:1,figures:[{id:'replacement-scene',name,pose:'standing',height_mm:height}]}))};}

for(const action of ['edit','restore','import'] as const){
 test(`a delayed photo analysis cannot replace a subsequent ${action}`,async({page})=>{
  const started=deferred(),release=deferred(),handled=deferred();
  await page.route('**/api/analyze',async route=>{started.resolve();await release.promise;try{await route.fulfill({json:referenceResponse});}catch{/* An explicit cancel aborts this request. */}finally{handled.resolve();}});
  await page.goto('/');await ready(page);
  await page.getByRole('button',{name:'From a photo'}).click();
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(photoFile);
  await started.promise;
  if(action==='edit')await page.getByLabel('Target overall height').fill('123');
  else if(action==='restore')await page.getByRole('button',{name:'Restore all defaults',exact:true}).click();
  else await page.locator('input[type=file][accept=".json,application/json"]').setInputFiles(projectFile('My replacement scene',115));
  const expectedHeight=action==='edit'?'123':action==='import'?'115':'90';
  const expectedName=action==='import'?'My replacement scene':'Little figure 1';
  await expect(page.getByLabel('Figure name')).toHaveValue(expectedName);
  await expect(page.getByLabel('Target overall height')).toHaveValue(expectedHeight);
  release.resolve();await handled.promise;
  if(action!=='edit')await ready(page);
  // Give an already-fulfilled response a browser task turn before checking persistence.
  await page.waitForTimeout(250);
  await expect(page.getByLabel('Figure name')).toHaveValue(expectedName);
  await expect(page.getByLabel('Target overall height')).toHaveValue(expectedHeight);
  await expect(page.getByRole('checkbox',{name:'Include Person 1'})).toHaveCount(0);
  await expect(page.getByText('Stale analysis warning')).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Generate figures',exact:true})).toBeEnabled();
  await page.waitForFunction(expected=>JSON.parse(localStorage.getItem('little-figures-studio:draft:v2')||'null')?.project.figures[0].height_mm===Number(expected),expectedHeight);
  const saved=await page.evaluate(()=>localStorage.getItem('little-figures-studio:draft:v2'));
  expect(saved).not.toContain('stale-analysis-person');
 });
}

test('validation describes its submitted snapshot when settings change during the request',async({page})=>{
 const started=deferred(),release=deferred();let submitted:any;
 await page.goto('/');await ready(page);
 await page.route('**/api/validate',async route=>{submitted=route.request().postDataJSON();started.resolve();await release.promise;await route.fulfill({json:{figures:[{id:submitted.figures[0].id,report:{status:'pass',checks:[{name:'Validation snapshot',status:'pass',detail:'Measured the submitted 90 mm figure.'}],dimensions:{width:50,depth:50,height:90},triangles:1000,components:1,caveat:'Snapshot fixture'}}]}});});
 await page.getByRole('button',{name:'Run checks',exact:true}).click();await started.promise;
 await page.getByLabel('Target overall height').fill('120');
 release.resolve();
 await expect(page.getByText('These checks describe the earlier settings. Generate and check again for your latest changes.')).toBeVisible();
 expect(submitted.figures[0].height_mm).toBe(90);
 await expect(page.getByText('Checks need updating for your current settings.')).toBeVisible();
 await expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeDisabled();
 await expect(page.getByLabel('Target overall height')).toHaveValue('120');
});

test('an earlier generation cannot overwrite an imported scene that finishes first',async({page})=>{
 const oldStarted=deferred(),releaseOld=deferred(),oldHandled=deferred();let oldId='';
 await page.goto('/');await ready(page);
 await page.route('**/api/jobs',async route=>{const submitted=route.request().postDataJSON();if(submitted.figures[0].height_mm!==100)return route.continue();oldId=submitted.figures[0].id;await route.fulfill({json:{id:'delayed-generation'}});});
 await page.route('**/api/jobs/delayed-generation',async route=>{oldStarted.resolve();await releaseOld.promise;try{await route.fulfill({json:{state:'ready',stage:'Older generation complete',result:{figures:[{id:oldId,name:'Old figure',vertices:[[0,0,0],[10,0,0],[0,10,0],[0,0,100]],faces:[[0,1,2],[0,1,3],[1,2,3],[2,0,3]],report:{status:'pass',checks:[],dimensions:{width:10,depth:10,height:100},triangles:4,components:1,caveat:'Old response fixture'}}]}}});}catch{/* Superseded generation polls are aborted. */}finally{oldHandled.resolve();}});
 await page.getByLabel('Target overall height').fill('100');
 await page.getByRole('button',{name:'Generate figures',exact:true}).click();await oldStarted.promise;
 await page.locator('input[type=file][accept=".json,application/json"]').setInputFiles(projectFile('Keep this final scene',130));
 await expect(page.getByLabel('Figure name')).toHaveValue('Keep this final scene');await ready(page);
 releaseOld.resolve();await oldHandled.promise;await page.waitForTimeout(250);
 await expect(page.getByRole('button',{name:'Download STL',exact:true})).toBeEnabled();
 await expect(page.getByLabel('Figure name')).toHaveValue('Keep this final scene');
 await expect(page.getByLabel('Target overall height')).toHaveValue('130');
 await expect(page.locator('.dimension-cards').getByText('130.0')).toBeVisible();
});
