import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { Maximize, Move, RotateCcw, Ruler, MousePointer2, Box } from 'lucide-react';
import type { Figure, MeshData } from './types';
import { download } from './types';
import { createPreviewHTML, type PreviewScene } from './sharing';

type Props={data:MeshData[];figures:Figure[];selected:string;onSelect:(id:string)=>void;onTransform:(id:string,t:Figure['transform'])=>void;onDimensions:(dims:{width:number;depth:number;height:number})=>void;busy:boolean;readOnly?:boolean};
export type ViewportHandle={exportGLB:()=>Promise<void>;exportPNG:()=>Promise<void>;exportTurntable:()=>Promise<void>;exportPreview:()=>Promise<void>};
type Stage={scene:THREE.Scene;camera:THREE.PerspectiveCamera;renderer:THREE.WebGLRenderer;orbit:OrbitControls;gizmo:TransformControls;root:THREE.Group;markers:THREE.Group;reset:()=>void};
const GROUP_BASE='__group_base__';
const FIGURE_COLOR=0xd5e69d;

function bounds(root:THREE.Group){root.updateMatrixWorld(true);const box=new THREE.Box3();root.children.filter(o=>o.visible).forEach(o=>box.expandByObject(o,true));if(box.isEmpty())throw new Error('Generate at least one included figure before exporting.');return {box,center:box.getCenter(new THREE.Vector3()),size:box.getSize(new THREE.Vector3()),radius:Math.max(box.getBoundingSphere(new THREE.Sphere()).radius,1)}}
function fitDistance(camera:THREE.PerspectiveCamera,radius:number){const vertical=THREE.MathUtils.degToRad(camera.fov)/2;const horizontal=Math.atan(Math.tan(vertical)*camera.aspect);return radius/Math.sin(Math.min(vertical,horizontal))*1.15}
function disposeGroup(group:THREE.Group){group.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose())}});group.clear()}
function previewScene(root:THREE.Group):PreviewScene{
 const b=bounds(root);const meshes:PreviewScene['meshes']=[];
 root.children.filter(o=>o.visible).forEach(object=>{if(!(object instanceof THREE.Mesh))return;const mesh=object as THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial|THREE.MeshStandardMaterial[]>;const p=mesh.geometry.getAttribute('position'),n=mesh.geometry.getAttribute('normal'),normalMatrix=new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld),allIndices=mesh.geometry.index?Array.from(mesh.geometry.index.array):Array.from({length:p.count},(_,i)=>i),materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];
  materials.forEach((material,materialIndex)=>{const groups=mesh.geometry.groups.filter(group=>(group.materialIndex??0)===materialIndex);const sourceIndices=mesh.geometry.groups.length?groups.flatMap(group=>allIndices.slice(group.start,group.start+group.count)):allIndices;if(!sourceIndices.length)return;const positions:number[]=[],normals:number[]=[],indices:number[]=[],remap=new Map<number,number>();
   sourceIndices.forEach(original=>{let index=remap.get(original);if(index===undefined){index=remap.size;remap.set(original,index);const v=new THREE.Vector3().fromBufferAttribute(p,original).applyMatrix4(mesh.matrixWorld),normal=new THREE.Vector3().fromBufferAttribute(n,original).applyMatrix3(normalMatrix).normalize();positions.push(...v.toArray().map(x=>Math.round(x*1000)/1000));normals.push(...normal.toArray().map(x=>Math.round(x*1000)/1000))}indices.push(index)});
   meshes.push({name:mesh.name,positions,normals,indices,color:`#${material.color.getHexString()}`});
  });
 });
 return {meshes,dimensions:{width:b.size.x,depth:b.size.y,height:b.size.z},center:b.center.toArray(),radius:b.radius};
}

async function recordTurntable(stage:Stage):Promise<Blob>{
 if(typeof MediaRecorder==='undefined'||typeof HTMLCanvasElement.prototype.captureStream!=='function')throw new Error('Video recording is unavailable in this browser. Download a PNG or the interactive HTML preview instead.');
 const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(type=>MediaRecorder.isTypeSupported(type));
 if(!mime)throw new Error('This browser cannot export WebM video. Try Chrome, Edge, or Firefox, or download the interactive preview.');
 const b=bounds(stage.root),scene=new THREE.Scene();scene.background=new THREE.Color('#e9ece3');scene.add(stage.root.clone(true));
 scene.add(new THREE.HemisphereLight(0xffffff,0x839075,2.7));const key=new THREE.DirectionalLight(0xfff9e9,4);key.position.copy(b.center).add(new THREE.Vector3(-b.radius,-b.radius,b.radius*2));key.target.position.copy(b.center);scene.add(key,key.target);const fill=new THREE.DirectionalLight(0xd9ead5,1.8);fill.position.copy(b.center).add(new THREE.Vector3(b.radius,b.radius,b.radius));fill.target.position.copy(b.center);scene.add(fill,fill.target);
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setSize(1280,720);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
 const camera=new THREE.PerspectiveCamera(35,1280/720,.1,b.radius*30);camera.up.set(0,0,1);const distance=fitDistance(camera,b.radius),stream=renderer.domElement.captureStream(30);
 let frame=0,timer=0;const chunks:Blob[]=[];
 try{return await new Promise<Blob>((resolve,reject)=>{let recorder:MediaRecorder;try{recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:6_000_000})}catch{reject(new Error('Your browser could not start video recording. Try downloading an image or interactive preview.'));return}
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  recorder.onerror=()=>reject(new Error('Video recording failed. Keep this tab visible and try again, or export an image.'));
  recorder.onstop=()=>{const blob=new Blob(chunks,{type:'video/webm'});if(blob.size)resolve(blob);else reject(new Error('The browser returned an empty video. Try an image or interactive preview.'))};
  const start=performance.now();const render=()=>{const progress=Math.min((performance.now()-start)/8000,1),angle=-1.03+progress*Math.PI*2;camera.position.set(b.center.x+distance*.86*Math.cos(angle),b.center.y+distance*.86*Math.sin(angle),b.center.z+distance*.51);camera.lookAt(b.center);renderer.render(scene,camera);frame=requestAnimationFrame(render)};render();recorder.start(250);timer=window.setTimeout(()=>{if(recorder.state!=='inactive')recorder.stop()},8100);
 })}finally{cancelAnimationFrame(frame);window.clearTimeout(timer);stream.getTracks().forEach(track=>track.stop());renderer.dispose()}
}

export const Viewport=forwardRef<ViewportHandle,Props>(function Viewport(props,ref){
 const host=useRef<HTMLDivElement>(null),current=useRef(props);current.current=props;
 const state=useRef<Stage|null>(null),exporting=useRef(false),fitSignature=useRef('');
 const [tool,setTool]=useState<'orbit'|'translate'|'rotate'|'measure'>('orbit'),toolRef=useRef(tool);toolRef.current=tool;
 const [distance,setDistance]=useState<number|null>(null),points=useRef<THREE.Vector3[]>([]),[webglError,setWebglError]=useState(''),[recording,setRecording]=useState(false);
 useImperativeHandle(ref,()=>({
  async exportGLB(){const s=state.current;if(!s)throw new Error('The 3D preview is unavailable. You can still export STL.');bounds(s.root);const exported=new THREE.Group();exported.add(s.root.clone(true));exported.rotation.x=-Math.PI/2;exported.scale.setScalar(.001);const result=await new GLTFExporter().parseAsync(exported,{binary:true,onlyVisible:true});if(!(result instanceof ArrayBuffer))throw new Error('The browser could not create the GLB file.');download(new Blob([result],{type:'model/gltf-binary'}),'little-figures.glb')},
  async exportPNG(){const s=state.current;if(!s)throw new Error('The 3D preview is unavailable. Enable hardware acceleration to save an image.');bounds(s.root);if(exporting.current)throw new Error('Another export is still running.');exporting.current=true;const helper=s.gizmo.getHelper(),helperVisible=helper.visible,markersVisible=s.markers.visible;helper.visible=false;s.markers.visible=false;try{s.renderer.render(s.scene,s.camera);const blob=await new Promise<Blob>((resolve,reject)=>s.renderer.domElement.toBlob(result=>result?resolve(result):reject(new Error('The browser could not create this image.')),'image/png'));download(blob,'little-figures.png')}finally{helper.visible=helperVisible;s.markers.visible=markersVisible;exporting.current=false}},
  async exportTurntable(){const s=state.current;if(!s)throw new Error('The 3D preview is unavailable. Enable hardware acceleration to record a video.');if(exporting.current)throw new Error('Another export is still running.');exporting.current=true;setRecording(true);try{download(await recordTurntable(s),'little-figures-turntable.webm')}finally{exporting.current=false;setRecording(false)}},
  async exportPreview(){const s=state.current;if(!s)throw new Error('Generate a 3D preview before sharing.');download(new Blob([createPreviewHTML(previewScene(s.root))],{type:'text/html;charset=utf-8'}),'little-figures-preview.html')},
 }));
 useEffect(()=>{
  if(!host.current)return;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#e9ece3');
  const camera=new THREE.PerspectiveCamera(35,1,.1,2500);camera.up.set(0,0,1);camera.position.set(145,-210,145);
  let renderer:THREE.WebGLRenderer;
  try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true})}catch{setWebglError('3D preview requires WebGL. Enable hardware acceleration or use a supported browser. STL export remains available.');return}
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
  host.current.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','Interactive 3D figurine preview. Drag to orbit; scroll to zoom. Use property fields for keyboard editing.');renderer.domElement.tabIndex=0;
  const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.set(0,0,42);orbit.enableDamping=true;orbit.minDistance=20;orbit.maxDistance=2000;orbit.maxPolarAngle=Math.PI*.49;
  scene.add(new THREE.HemisphereLight(0xffffff,0x839075,2.7));
  const light=new THREE.DirectionalLight(0xfff9e9,4);light.position.set(-70,-100,190);light.castShadow=true;light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-180;light.shadow.camera.right=180;light.shadow.camera.top=180;light.shadow.camera.bottom=-180;light.shadow.normalBias=.08;scene.add(light);scene.add(light.target);
  const fill=new THREE.DirectionalLight(0xd9ead5,1.8);fill.position.set(100,50,80);scene.add(fill);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(2000,2000),new THREE.MeshStandardMaterial({color:0xe9ece3,roughness:1}));floor.position.z=-.35;floor.receiveShadow=true;scene.add(floor);
  const grid=new THREE.GridHelper(1000,100,0xb8c1b1,0xcfd6c8);grid.rotation.x=Math.PI/2;grid.position.z=-.25;(grid.material as THREE.Material).transparent=true;(grid.material as THREE.Material).opacity=.48;scene.add(grid);
  const root=new THREE.Group();scene.add(root);const markers=new THREE.Group();scene.add(markers);
  const gizmo=new TransformControls(camera,renderer.domElement);gizmo.setSpace('world');gizmo.setSize(.75);scene.add(gizmo.getHelper());
  gizmo.addEventListener('dragging-changed',e=>{orbit.enabled=!e.value});
  gizmo.addEventListener('mouseUp',()=>{const object=gizmo.object;if(object){const f=current.current.figures.find(f=>f.id===object.userData.id);if(f)current.current.onTransform(f.id,{...f.transform,x:Math.round(object.position.x*10)/10,y:Math.round(object.position.y*10)/10,rotation:Math.round(THREE.MathUtils.radToDeg(object.rotation.z))})}});
  const reset=()=>{let b;try{b=bounds(root)}catch{return}const distance=fitDistance(camera,b.radius);camera.position.copy(b.center).add(new THREE.Vector3(.52,-.76,.5).normalize().multiplyScalar(distance));camera.far=Math.max(2500,distance+b.radius*12);camera.updateProjectionMatrix();orbit.target.copy(b.center);orbit.minDistance=Math.max(5,b.radius*.4);orbit.maxDistance=Math.max(600,b.radius*15);const extent=Math.max(b.radius*1.5,100);light.shadow.camera.left=-extent;light.shadow.camera.right=extent;light.shadow.camera.top=extent;light.shadow.camera.bottom=-extent;light.shadow.camera.far=Math.max(500,b.radius*8);light.shadow.camera.updateProjectionMatrix();light.position.copy(b.center).add(new THREE.Vector3(-b.radius,-b.radius,b.radius*2));light.target.position.copy(b.center);orbit.update()};
  state.current={scene,camera,renderer,orbit,gizmo,root,markers,reset};
  const raycaster=new THREE.Raycaster(),down={x:0,y:0};
  const pointerdown=(e:PointerEvent)=>{down.x=e.clientX;down.y=e.clientY};
  const pointerup=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>5||gizmo.dragging)return;
   const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
   const hits=raycaster.intersectObjects(root.children.filter(o=>o.visible),true),hit=toolRef.current==='measure'?hits[0]:hits.find(h=>h.object.userData.id!==GROUP_BASE);if(!hit)return;
   if(toolRef.current==='measure'){
    if(points.current.length===2){points.current=[];disposeGroup(markers);setDistance(null)}points.current.push(hit.point.clone());
    const dot=new THREE.Mesh(new THREE.SphereGeometry(.9,12,12),new THREE.MeshBasicMaterial({color:0xe0844c,depthTest:false}));dot.position.copy(hit.point);dot.renderOrder=10;markers.add(dot);
    if(points.current.length===2){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.current),new THREE.LineBasicMaterial({color:0xb45d30,depthTest:false}));line.renderOrder=10;markers.add(line);setDistance(points.current[0].distanceTo(points.current[1]))}
   }else current.current.onSelect(hit.object.userData.id);
  };
  renderer.domElement.addEventListener('pointerdown',pointerdown);renderer.domElement.addEventListener('pointerup',pointerup);
  const resize=new ResizeObserver(()=>{if(!host.current)return;const {width,height}=host.current.getBoundingClientRect();if(!width||!height)return;const oldAspect=camera.aspect;camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height);if(Math.abs(oldAspect-camera.aspect)>.15)reset()});resize.observe(host.current);
  let frame=0;const render=()=>{frame=requestAnimationFrame(render);orbit.update();renderer.render(scene,camera)};render();
  return ()=>{cancelAnimationFrame(frame);resize.disconnect();orbit.dispose();gizmo.dispose();scene.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose())}});renderer.dispose();renderer.domElement.remove();state.current=null};
 },[]);
 useEffect(()=>{const s=state.current;if(!s)return;s.gizmo.detach();disposeGroup(s.root);fitSignature.current='';
  props.data.forEach(d=>{const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(d.vertices.flat(),3));geometry.setIndex(d.faces.flat());geometry.computeVertexNormals();geometry.computeBoundingBox();const mat=new THREE.MeshStandardMaterial({color:d.id===GROUP_BASE?0xbdcda3:FIGURE_COLOR,roughness:.68,metalness:0});let material:THREE.MeshStandardMaterial|THREE.MeshStandardMaterial[]=mat;
   if(d.face_groups?.length===d.faces.length){const faceGroups=d.face_groups,body:number[]=[],face:number[]=[];d.faces.forEach((triangle,i)=>(faceGroups[i]===1?face:body).push(...triangle));geometry.setIndex([...body,...face]);if(body.length)geometry.addGroup(0,body.length,0);if(face.length)geometry.addGroup(body.length,face.length,1);material=[mat,new THREE.MeshStandardMaterial({color:0x35422a,roughness:.85,metalness:0})]}
   const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=true;mesh.receiveShadow=true;mesh.userData.id=d.id;mesh.name=d.name;s.root.add(mesh)});
  points.current=[];disposeGroup(s.markers);setDistance(null);
 },[props.data]);
 useEffect(()=>{const s=state.current;if(!s)return;points.current=[];disposeGroup(s.markers);setDistance(null);
  s.root.children.forEach(o=>{if(o.userData.id===GROUP_BASE){o.name='Shared pedestal';o.visible=true;o.position.set(0,0,0);o.rotation.set(0,0,0);o.scale.setScalar(1);return}const f=props.figures.find(f=>f.id===o.userData.id);if(!f){o.visible=false;return}o.name=f.name;o.visible=f.included;o.position.set(f.transform.x,f.transform.y,0);o.rotation.set(0,0,THREE.MathUtils.degToRad(f.transform.rotation));o.scale.setScalar(f.transform.scale);const material=(o as THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial|THREE.MeshStandardMaterial[]>).material;(Array.isArray(material)?material:[material]).forEach(m=>m.emissive.set(o.userData.id===props.selected?0x080d03:0x000000))});
  s.root.updateMatrixWorld(true);const selected=s.root.children.find(o=>o.userData.id===props.selected&&o.visible&&o.userData.id!==GROUP_BASE);
  if(props.readOnly){try{const {size:v}=bounds(s.root);props.onDimensions({width:v.x,depth:v.y,height:v.z})}catch{props.onDimensions({width:0,depth:0,height:0})}}else if(selected){const box=new THREE.Box3().setFromObject(selected,true),v=box.getSize(new THREE.Vector3());props.onDimensions({width:v.x,depth:v.y,height:v.z})}else props.onDimensions({width:0,depth:0,height:0});
  s.gizmo.detach();if(!props.readOnly&&selected&&(tool==='translate'||tool==='rotate')){s.gizmo.attach(selected);s.gizmo.setMode(tool==='translate'?'translate':'rotate');s.gizmo.showX=tool==='translate';s.gizmo.showY=tool==='translate';s.gizmo.showZ=tool==='rotate'}
  const signature=s.root.children.filter(o=>o.visible).map(o=>o.userData.id).join('|');if(signature&&signature!==fitSignature.current){fitSignature.current=signature;s.reset()}
 },[props.figures,props.selected,props.data,props.readOnly,tool]);
 const view=(which:string)=>{const s=state.current;if(!s)return;if(which==='perspective'){s.reset();return}let b;try{b=bounds(s.root)}catch{return}const distance=fitDistance(s.camera,b.radius);s.orbit.target.copy(b.center);s.camera.position.copy(b.center).add(which==='front'?new THREE.Vector3(0,-distance,0):which==='side'?new THREE.Vector3(distance,0,0):new THREE.Vector3(0,-.01,distance));s.orbit.update()};
 return <div className="workspace"><div ref={host} className="canvas-host"/>{webglError&&<div className="webgl-error">{webglError}</div>}
  <div className="viewport-top"><span className="scene-label"><span className="live-dot"/> PERSPECTIVE VIEW</span><span className="unit-label">Z up · millimeters</span></div>
  <div className="view-tools" role="toolbar" aria-label="Viewport tools">{([['orbit',MousePointer2,'Select / orbit'],['translate',Move,'Move figure'],['rotate',RotateCcw,'Rotate figure'],['measure',Ruler,'Measure two surface points']] as const).filter(([id])=>!props.readOnly||id==='orbit'||id==='measure').map(([id,Icon,label])=><button key={id} title={label} aria-label={label} aria-pressed={tool===id} className={tool===id?'active':''} onClick={()=>setTool(id)}><Icon size={18}/></button>)}<span/><button title="Fit scene" aria-label="Fit scene" onClick={()=>view('perspective')}><Maximize size={18}/></button></div>
  <div className="camera-views">{['front','side','top'].map(v=><button key={v} onClick={()=>view(v)}>{v}</button>)}<button aria-label="Perspective camera" onClick={()=>view('perspective')}><Box size={14}/></button></div>
  <div className="viewport-bottom"><span><i/>10 mm grid</span><span>{tool==='measure'?(distance===null?'Click two points on a surface':`${distance.toFixed(2)} mm · click to restart`):'Drag to orbit · scroll to zoom · right-drag to pan'}</span></div>
  {(props.busy||recording)&&<div className="viewport-busy"><span className="spinner"/>{recording?'Recording your turntable… keep this tab visible (8 seconds).':'Sculpting your little companions…'}</div>}
 </div>
});
