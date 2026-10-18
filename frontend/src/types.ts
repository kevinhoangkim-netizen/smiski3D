export type Landmark={x:number;y:number;z:number;visibility:number};
export type Pose='standing'|'seated'|'reaching'|'leaning'|'crouching'|'arms-raised'|'photo';
export type Accessory='hat'|'bouquet'|'veil'|'bowtie';
export type Appearance={outfit:'none'|'casual'|'suit'|'wedding-dress';accessories:Accessory[];hat_style:'cap'|'top-hat'|'wide-brim';bouquet_hand:'left'|'right'|'both'};
export type AppearanceDetection={status:'ready'|'unavailable'|'uncertain';model?:string;confidence?:number|null;warnings:string[]};
export type Figure={id:string;name:string;included:boolean;pose:Pose;landmarks:Landmark[]|null;photo_aspect:number;height_mm:number;head_size:number;base:{shape:'none'|'round'|'rectangle';margin_mm:number;thickness_mm:number};transform:{x:number;y:number;rotation:number;scale:number};appearance:Appearance;appearance_source:'manual'|'automatic'|'unavailable';appearance_confidence:number|null;analysis_warnings:string[]};
export type Project={version:1;figures:Figure[];profile:'resin'|'fdm';minimum_feature_mm:number;shared_base:{shape:'none'|'round'|'rectangle';margin_mm:number;thickness_mm:number;inscription:string;text_size_mm:number};occasion:'everyday'|'wedding'|'family'|'graduation'|'friendship'};
export type PhotoSubject={id:string;name:string;landmarks:Landmark[];bbox:number[];confidence:number;warnings:string[];appearance?:Appearance;appearance_detection?:AppearanceDetection;suggested_transform?:{x:number;y:number;scale:number};suggested_height_mm?:number;photo_aspect?:number};
export type Check={name:string;status:'pass'|'warning'|'fail';detail:string};
export type Report={status:'pass'|'warning'|'fail';checks:Check[];dimensions:{width:number;depth:number;height:number};triangles:number;components:number;caveat:string};
export type MeshData={id:string;name:string;vertices:number[][];faces:number[][];face_groups?:number[];report:Report};
export const poses:Pose[]=['standing','seated','reaching','leaning','crouching','arms-raised'];
export const poseLabels:Record<Pose,string>={'standing':'Just standing','seated':'Taking a seat','reaching':'A little reach','leaning':'Leaning in','crouching':'Crouching','arms-raised':'Hands up!','photo':'From your photo'};
export const defaultAppearance=():Appearance=>({outfit:'none',accessories:[],hat_style:'top-hat',bouquet_hand:'both'});
export function newFigure(index=0):Figure{return {id:crypto.randomUUID(),name:`Little figure ${index+1}`,included:true,pose:index?'seated':'standing',landmarks:null,photo_aspect:1,height_mm:90,head_size:1,base:{shape:'round',margin_mm:5,thickness_mm:3},transform:{x:index*65,y:0,rotation:0,scale:1},appearance:defaultAppearance(),appearance_source:'manual',appearance_confidence:null,analysis_warnings:[]};}
export const initialProject=():Project=>({version:1,figures:[newFigure()],profile:'resin',minimum_feature_mm:1.2,shared_base:{shape:'none',margin_mm:8,thickness_mm:3,inscription:'',text_size_mm:4},occasion:'everyday'});
export function normalizeProject(value:Project):Project{const defaults=initialProject();return {...defaults,...value,shared_base:{...defaults.shared_base,...value.shared_base},figures:value.figures.map((f,i)=>({...newFigure(i),...f,appearance:{...defaultAppearance(),...f.appearance}}))};}
export async function api<T>(url:string,options?:RequestInit):Promise<T>{
 const r=await fetch(url,options);if(!r.ok){let msg=`Request failed (${r.status})`;try{const body=await r.json();msg=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)}catch{}throw new Error(msg)}return r.json();
}
export function jsonPost(value:unknown):RequestInit{return {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}}
export function download(blob:Blob,name:string){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
