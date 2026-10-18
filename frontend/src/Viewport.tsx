import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { Maximize, Move, RotateCcw, Ruler, MousePointer2, Camera, Box } from 'lucide-react';
import type { Figure, MeshData } from './types';
import { download } from './types';
type Props={data:MeshData[];figures:Figure[];selected:string;onSelect:(id:string)=>void;onTransform:(id:string,t:Figure['transform'])=>void;onDimensions:(dims:{width:number;depth:number;height:number})=>void;busy:boolean};
export type ViewportHandle={exportGLB:()=>void};
export const Viewport=forwardRef<ViewportHandle,Props>(function Viewport(props,ref){
 const host=useRef<HTMLDivElement>(null);const current=useRef(props);current.current=props;
 const state=useRef<{scene:THREE.Scene;camera:THREE.PerspectiveCamera;renderer:THREE.WebGLRenderer;orbit:OrbitControls;gizmo:TransformControls;root:THREE.Group;markers:THREE.Group;reset:()=>void}|null>(null);
 const [tool,setTool]=useState<'orbit'|'translate'|'rotate'|'measure'>('orbit');const toolRef=useRef(tool);toolRef.current=tool;
 const [distance,setDistance]=useState<number|null>(null);const points=useRef<THREE.Vector3[]>([]);const [webglError,setWebglError]=useState('');
 useImperativeHandle(ref,()=>({exportGLB(){const s=state.current;if(!s)return;const exported=new THREE.Group();exported.add(s.root.clone(true));exported.rotation.x=-Math.PI/2;exported.scale.setScalar(.001);new GLTFExporter().parse(exported,(result)=>{if(result instanceof ArrayBuffer) download(new Blob([result],{type:'model/gltf-binary'}),'little-figures.glb')},console.error,{binary:true});}}));
 useEffect(()=>{
  if(!host.current)return;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#e9ece3');
  const camera=new THREE.PerspectiveCamera(35,1,.1,2500);camera.up.set(0,0,1);camera.position.set(145,-210,145);
  let renderer:THREE.WebGLRenderer;
  try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:false})}catch{setWebglError('3D preview requires WebGL. Enable hardware acceleration or use a supported browser. STL export remains available.');return}
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
  host.current.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','Interactive 3D figurine preview. Drag to orbit; scroll to zoom. Use property fields for keyboard editing.');renderer.domElement.tabIndex=0;
  const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.set(0,0,42);orbit.enableDamping=true;orbit.minDistance=50;orbit.maxDistance=600;orbit.maxPolarAngle=Math.PI*.49;
  scene.add(new THREE.HemisphereLight(0xffffff,0x839075,2.7));
  const light=new THREE.DirectionalLight(0xfff9e9,4);light.position.set(-70,-100,190);light.castShadow=true;light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-180;light.shadow.camera.right=180;light.shadow.camera.top=180;light.shadow.camera.bottom=-180;light.shadow.normalBias=.08;scene.add(light);
  const fill=new THREE.DirectionalLight(0xd9ead5,1.8);fill.position.set(100,50,80);scene.add(fill);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(2000,2000),new THREE.MeshStandardMaterial({color:0xe9ece3,roughness:1}));floor.position.z=-.35;floor.receiveShadow=true;scene.add(floor);
  const grid=new THREE.GridHelper(500,50,0xb8c1b1,0xcfd6c8);grid.rotation.x=Math.PI/2;grid.position.z=-.25;(grid.material as THREE.Material).transparent=true;(grid.material as THREE.Material).opacity=.48;scene.add(grid);
  const root=new THREE.Group();scene.add(root);const markers=new THREE.Group();scene.add(markers);
  const gizmo=new TransformControls(camera,renderer.domElement);gizmo.setSpace('world');gizmo.setSize(.75);scene.add(gizmo.getHelper());
  gizmo.addEventListener('dragging-changed',e=>{orbit.enabled=!e.value});
  gizmo.addEventListener('mouseUp',()=>{const object=gizmo.object;if(object){const f=current.current.figures.find(f=>f.id===object.userData.id);if(f)current.current.onTransform(f.id,{...f.transform,x:Math.round(object.position.x*10)/10,y:Math.round(object.position.y*10)/10,rotation:Math.round(THREE.MathUtils.radToDeg(object.rotation.z))})}});
  const reset=()=>{camera.position.set(145,-210,145);orbit.target.set(0,0,42);orbit.update()};
  state.current={scene,camera,renderer,orbit,gizmo,root,markers,reset};
  const raycaster=new THREE.Raycaster();const down={x:0,y:0};
  const pointerdown=(e:PointerEvent)=>{down.x=e.clientX;down.y=e.clientY};
  const pointerup=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>5||gizmo.dragging)return;
    const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
    const hit=raycaster.intersectObjects(root.children,true)[0];if(!hit)return;
    if(toolRef.current==='measure'){
      if(points.current.length===2){points.current=[];markers.clear();setDistance(null)}points.current.push(hit.point.clone());
      const dot=new THREE.Mesh(new THREE.SphereGeometry(.9,12,12),new THREE.MeshBasicMaterial({color:0xe0844c,depthTest:false}));dot.position.copy(hit.point);dot.renderOrder=10;markers.add(dot);
      if(points.current.length===2){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.current),new THREE.LineBasicMaterial({color:0xb45d30,depthTest:false}));line.renderOrder=10;markers.add(line);setDistance(points.current[0].distanceTo(points.current[1]))}
    }else current.current.onSelect(hit.object.userData.id);
  };
  renderer.domElement.addEventListener('pointerdown',pointerdown);renderer.domElement.addEventListener('pointerup',pointerup);
  const resize=new ResizeObserver(()=>{if(!host.current)return;const {width,height}=host.current.getBoundingClientRect();camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height)});resize.observe(host.current);
  let frame=0;const render=()=>{frame=requestAnimationFrame(render);orbit.update();renderer.render(scene,camera)};render();
  return ()=>{cancelAnimationFrame(frame);resize.disconnect();orbit.dispose();gizmo.dispose();scene.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();const ms=Array.isArray(o.material)?o.material:[o.material];ms.forEach(m=>m.dispose())}});renderer.dispose();renderer.domElement.remove();state.current=null};
 },[]);
 useEffect(()=>{const s=state.current;if(!s)return;s.gizmo.detach();s.root.children.forEach(o=>{const m=o as THREE.Mesh;m.geometry.dispose();(m.material as THREE.Material).dispose()});s.root.clear();
 props.data.forEach((d,i)=>{const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(d.vertices.flat(),3));geometry.setIndex(d.faces.flat());geometry.computeVertexNormals();geometry.computeBoundingBox();const mat=new THREE.MeshStandardMaterial({color:[0xb2c996,0xdcc3a1,0xa7c3c6][i%3],roughness:.6,metalness:0});const mesh=new THREE.Mesh(geometry,mat);mesh.castShadow=true;mesh.receiveShadow=true;mesh.userData.id=d.id;mesh.name=d.name;s.root.add(mesh)});
 points.current=[];s.markers.clear();setDistance(null);
 },[props.data]);
 useEffect(()=>{const s=state.current;if(!s)return;points.current=[];s.markers.clear();setDistance(null);
 s.root.children.forEach(o=>{const f=props.figures.find(f=>f.id===o.userData.id);if(!f){o.visible=false;return}o.visible=f.included;o.position.set(f.transform.x,f.transform.y,0);o.rotation.set(0,0,THREE.MathUtils.degToRad(f.transform.rotation));o.scale.setScalar(f.transform.scale);const m=o as THREE.Mesh;(m.material as THREE.MeshStandardMaterial).emissive.set(o.userData.id===props.selected?0x0a1105:0x000000)});
 s.root.updateMatrixWorld(true);const selected=s.root.children.find(o=>o.userData.id===props.selected&&o.visible);
 if(selected){const box=new THREE.Box3().setFromObject(selected,true);const v=box.getSize(new THREE.Vector3());props.onDimensions({width:v.x,depth:v.y,height:v.z})}else props.onDimensions({width:0,depth:0,height:0});
 s.gizmo.detach();if(selected&&(tool==='translate'||tool==='rotate')){s.gizmo.attach(selected);s.gizmo.setMode(tool==='translate'?'translate':'rotate');s.gizmo.showX=tool==='translate';s.gizmo.showY=tool==='translate';s.gizmo.showZ=tool==='rotate'}
 },[props.figures,props.selected,props.data,tool]);
 const view=(which:string)=>{const s=state.current;if(!s)return;if(which==='perspective'){s.reset();return}const t=s.orbit.target.clone();s.camera.position.copy(t.clone().add(which==='front'?new THREE.Vector3(0,-260,0):which==='side'?new THREE.Vector3(260,0,0):new THREE.Vector3(0,-.01,260)));s.orbit.update()};
 return <div className="workspace"><div ref={host} className="canvas-host"/>{webglError&&<div className="webgl-error">{webglError}</div>}
 <div className="viewport-top"><span className="scene-label"><span className="live-dot"/> PERSPECTIVE VIEW</span><span className="unit-label">Z up · millimeters</span></div>
 <div className="view-tools" role="toolbar" aria-label="Viewport tools">{([['orbit',MousePointer2,'Select / orbit'],['translate',Move,'Move figure'],['rotate',RotateCcw,'Rotate figure'],['measure',Ruler,'Measure two surface points']] as const).map(([id,Icon,label])=><button key={id} title={label} aria-label={label} aria-pressed={tool===id} className={tool===id?'active':''} onClick={()=>setTool(id)}><Icon size={18}/></button>)}<span/><button title="Reset camera" aria-label="Reset camera" onClick={()=>view('perspective')}><Maximize size={18}/></button></div>
 <div className="camera-views">{['front','side','top'].map(v=><button key={v} onClick={()=>view(v)}>{v}</button>)}<button aria-label="Perspective camera" onClick={()=>view('perspective')}><Box size={14}/></button></div>
 <div className="viewport-bottom"><span><i/>10 mm grid</span><span>{tool==='measure'?(distance===null?'Click two points on a figure':`${distance.toFixed(2)} mm · click to restart`):'Drag to orbit · scroll to zoom · right-drag to pan'}</span></div>
 {props.busy&&<div className="viewport-busy"><span className="spinner"/> Sculpting your little companion…</div>}
 </div>
});
