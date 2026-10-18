export type Landmark={x:number;y:number;z:number;visibility:number};
export type Pose='standing'|'seated'|'reaching'|'leaning'|'crouching'|'arms-raised'|'photo';
export type Figure={id:string;name:string;included:boolean;pose:Pose;landmarks:Landmark[]|null;height_mm:number;head_size:number;base:{shape:'none'|'round'|'rectangle';margin_mm:number;thickness_mm:number};transform:{x:number;y:number;rotation:number;scale:number}};
export type Project={version:1;figures:Figure[];profile:'resin'|'fdm';minimum_feature_mm:number};
export type Check={name:string;status:'pass'|'warning'|'fail';detail:string};
export type Report={status:'pass'|'warning'|'fail';checks:Check[];dimensions:{width:number;depth:number;height:number};triangles:number;components:number;caveat:string};
export type MeshData={id:string;name:string;vertices:number[][];faces:number[][];report:Report};
export const poses:Pose[]=['standing','seated','reaching','leaning','crouching','arms-raised'];
export const poseLabels:Record<Pose,string>={'standing':'Just standing','seated':'Taking a seat','reaching':'A little reach','leaning':'Leaning in','crouching':'Crouching','arms-raised':'Hands up!','photo':'From your photo'};
export function newFigure(index=0):Figure{return {id:crypto.randomUUID(),name:`Little figure ${index+1}`,included:true,pose:index?'seated':'standing',landmarks:null,height_mm:90,head_size:1,base:{shape:'round',margin_mm:5,thickness_mm:3},transform:{x:index*65,y:0,rotation:0,scale:1}};}
export const initialProject=():Project=>({version:1,figures:[newFigure()],profile:'resin',minimum_feature_mm:1.2});
export async function api<T>(url:string,options?:RequestInit):Promise<T>{
 const r=await fetch(url,options);if(!r.ok){let msg=`Request failed (${r.status})`;try{const body=await r.json();msg=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)}catch{}throw new Error(msg)}return r.json();
}
export function jsonPost(value:unknown):RequestInit{return {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}}
export function download(blob:Blob,name:string){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
