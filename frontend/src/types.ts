export type Landmark={x:number;y:number;z:number;visibility:number};
export type Pose='standing'|'seated'|'reaching'|'leaning'|'crouching'|'arms-raised'|'photo';
export type Accessory='hat'|'bouquet'|'veil'|'bowtie'|'bag'|'scarf'|'glasses';
export type Garment={upper:'unknown'|'bare'|'shirt'|'jacket'|'dress';lower:'unknown'|'bare'|'pants'|'shorts'|'skirt'|'dress';sleeve_length:number;hem_length:number;skirt_flare:number;trouser_length:number;headwear:'none'|'cap'|'brim'|'beanie'|'unknown';inferred:boolean;estimated_fields?:('upper'|'lower'|'sleeve_length'|'hem_length'|'trouser_length'|'skirt_flare')[];estimation_notes?:string[]};
export type Appearance={garment?:Garment;garment_evidence?:Record<string,unknown>|null;outfit:'none'|'casual'|'suit'|'wedding-dress';accessories:Accessory[];hat_style:'cap'|'top-hat'|'wide-brim'|'beanie';bouquet_hand:'left'|'right'|'both'};
export type AppearanceDetection={status:'ready'|'unavailable'|'uncertain';model?:string;confidence?:number|null;warnings:string[];summary?:string;unsupported_items?:string[];blocking?:boolean};
export type Figure={id:string;name:string;included:boolean;pose:Pose;landmarks:Landmark[]|null;photo_aspect:number;height_mm:number;head_size:number;base:{shape:'none'|'round'|'rectangle';margin_mm:number;thickness_mm:number};transform:{x:number;y:number;rotation:number;scale:number};appearance:Appearance;appearance_source:'manual'|'automatic'|'unavailable';appearance_confidence:number|null;analysis_warnings:string[];analysis_blockers?:string[]};
export type Project={version:1;photo_review?:{detected_people:number;unresolved_people:number;confirmed:boolean}|null;figures:Figure[];profile:'resin'|'fdm';minimum_feature_mm:number;shared_base:{shape:'none'|'round'|'rectangle';margin_mm:number;thickness_mm:number;inscription:string;text_size_mm:number};occasion:'everyday'|'wedding'|'family'|'graduation'|'friendship'};
export type PhotoSubject={id:string;name:string;landmarks:Landmark[];bbox:number[];confidence:number;warnings:string[];appearance?:Appearance;appearance_detection?:AppearanceDetection;suggested_transform?:{x:number;y:number;scale:number};suggested_height_mm?:number;photo_aspect?:number};
export type PhotoAnalysis={subjects:PhotoSubject[];detected_people:number;unresolved_regions:number[][];warnings:string[];detection:{available:boolean;model?:string;detail?:string};overflow:boolean};
export type Check={name:string;status:'pass'|'warning'|'fail';detail:string};
export type Report={status:'pass'|'warning'|'fail';checks:Check[];dimensions:{width:number;depth:number;height:number};triangles:number;components:number;caveat:string};
export type MeshData={id:string;name:string;vertices:number[][];faces:number[][];face_groups?:number[];report:Report};
export const poses:Pose[]=['standing','seated','reaching','leaning','crouching','arms-raised'];
export const poseLabels:Record<Pose,string>={'standing':'Just standing','seated':'Taking a seat','reaching':'A little reach','leaning':'Leaning in','crouching':'Crouching','arms-raised':'Hands up!','photo':'From your photo'};
export const defaultAppearance=():Appearance=>({outfit:'none',accessories:[],hat_style:'top-hat',bouquet_hand:'both'});
export function newFigure(index=0):Figure{return {id:crypto.randomUUID(),name:`Person ${index+1}`,included:true,pose:'photo',landmarks:null,photo_aspect:1,height_mm:90,head_size:1,base:{shape:'round',margin_mm:5,thickness_mm:3},transform:{x:0,y:0,rotation:0,scale:1},appearance:defaultAppearance(),appearance_source:'unavailable',appearance_confidence:null,analysis_warnings:[],analysis_blockers:[]};}
export const initialProject=():Project=>({version:1,photo_review:null,figures:[],profile:'resin',minimum_feature_mm:1.2,shared_base:{shape:'none',margin_mm:8,thickness_mm:3,inscription:'',text_size_mm:4},occasion:'everyday'});
export function normalizeProject(value:Project):Project{const defaults=initialProject();return {...defaults,...value,shared_base:{...defaults.shared_base,...value.shared_base},figures:value.figures.map((f,i)=>({...newFigure(i),...f,appearance:{...defaultAppearance(),...f.appearance}}))};}
export async function api<T>(url:string,options?:RequestInit):Promise<T>{
 const r=await fetch(url,options);if(!r.ok){let msg=`Request failed (${r.status})`;try{const body=await r.json();msg=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)}catch{}throw new Error(msg)}return r.json();
}
export function jsonPost(value:unknown):RequestInit{return {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}}
export function download(blob:Blob,name:string){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
