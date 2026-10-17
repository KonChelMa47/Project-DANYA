import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{STLLoader}from'../../outputs/danya-simulator/node_modules/three/examples/jsm/loaders/STLLoader.js';import{MeshBVH}from'../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';import{obs}from'./route-check.mjs';import{around,pivot,pitchPivot}from'./motion.mjs';
import {ConvexGeometry}from'../../outputs/danya-simulator/node_modules/three/examples/jsm/geometries/ConvexGeometry.js';
for(const o of obs.filter(o=>(o.name.startsWith('motor-')||o.name==='projector'))){o.actual=o.g;const points=[];for(let i=0;i<o.g.attributes.position.count;i++)points.push(new T.Vector3().fromBufferAttribute(o.g.attributes.position,i));o.g=new ConvexGeometry(points);o.g.boundsTree=new MeshBVH(o.g);}
const root='outputs/danya-simulator/assets/buildable-bust-v10';
for(const[name,stage]of[['yaw-motor-rear-stand',0],['mirror-head-cradle',3]]){let b=fs.readFileSync(root+'/'+name+'.stl'),g=new STLLoader().parse(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));g.boundsTree=new MeshBVH(g);obs.push({name,stage,g});}
function matrices(v){const[y,p,r]=v;return[new T.Matrix4(),around({y,p:0,r:0}),around({y,p,r:0}),around({y,p,r})];}
function radius(o){let max=0,p=new T.Vector3(),a=o.g.attributes.position;for(let i=0;i<a.count;i++){p.fromBufferAttribute(a,i);max=Math.max(max,p.distanceTo(pivot),p.distanceTo(pitchPivot));}return max+48;}
const rows=[];
for(const name of['yaw-motor-rear-stand','mirror-head-cradle']){
 const a=obs.find(o=>o.name===name);
 for(const b of obs.filter(o=>['motor-yaw','motor-pitch','motor-roll','projector','head','face','body','mirror-glass','foldMirror-glass'].includes(o.name)&&(!process.argv.includes('--critical')||o.name.startsWith('motor-')||o.name==='body')).sort((a,b)=>Number(b.name.startsWith('motor-'))-Number(a.name.startsWith('motor-')))){
  console.log('START',a.name,b.name);
  const low=Math.min(a.stage,b.stage),high=Math.max(a.stage,b.stage),R=radius(a.stage>b.stage?a:b),active=[0,1,2].filter(i=>i>=low&&i<high);
  let box=[[-30,30],[0,12],[-6,6]].map((v,i)=>active.includes(i)?v:[0,0]),queue=[box],calls=0,leaves=0,minSample=Infinity,guaranteed=Infinity,status='certified_surface_clearance',failure=null;const margin=1.0;
  while(queue.length){const cell=queue.pop(),center=cell.map(([l,h])=>(l+h)/2),ms=matrices(center),m=ms[a.stage].clone().invert().multiply(ms[b.stage]);
   const bound=cell.reduce((sum,[l,h])=>sum+2*R*Math.sin((h-l)*Math.PI/720),0);
   if(!a.g.boundingBox)a.g.computeBoundingBox();if(!b.g.boundingBox)b.g.computeBoundingBox();const bb=b.g.boundingBox.clone().applyMatrix4(m),aa=a.g.boundingBox;let boxDistance=0;for(const axis of ['x','y','z'])boxDistance+=Math.max(0,aa.min[axis]-bb.max[axis],bb.min[axis]-aa.max[axis])**2;boxDistance=Math.sqrt(boxDistance);
   if(boxDistance>bound+margin){leaves++;guaranteed=Math.min(guaranteed,boxDistance-bound);continue;}
   if(bound>5){let axis=0;for(let i=1;i<3;i++)if(cell[i][1]-cell[i][0]>cell[axis][1]-cell[axis][0])axis=i;for(let side of[0,1]){const child=cell.map(v=>v.slice());child[axis][side]=center[axis];queue.push(child);}continue;}
   const intersects=a.g.boundsTree.intersectsGeometry(b.g,m);const c=intersects?{distance:0}:a.g.boundsTree.closestPointToGeometry(b.g,m,{},{},bound+margin,bound+margin);calls++;const distance=c?.distance??Infinity;minSample=Math.min(minSample,distance);
   if(distance<margin){status=distance<1e-5?'conservative_envelope_contact':'clearance_below_margin';failure={angles:center,distance};break;}

   if(distance>bound+margin){leaves++;guaranteed=Math.min(guaranteed,Number.isFinite(distance)?distance-bound:margin);continue;}
   let axis=0;for(let i=1;i<3;i++)if(cell[i][1]-cell[i][0]>cell[axis][1]-cell[axis][0])axis=i;
   if(calls>30000||cell[axis][1]-cell[axis][0]<.02){status='unresolved';failure={angles:center,distance,bound};break;}
   for(let side of[0,1]){let child=cell.map(v=>v.slice());child[axis][side]=center[axis];queue.push(child);}
  }
  rows.push({a:a.name,b:b.name,status,calls,leaves,closestWitnessMM:Number.isFinite(minSample)?minSample:null,guaranteedSurfaceClearanceMM:status==='certified_surface_clearance'?guaranteed:null,failure});console.log(JSON.stringify(rows.at(-1)));
  fs.writeFileSync(root+'/continuous-critical-clearance.json',JSON.stringify({scope:'Only named pairs. Yaw[-30,30],pitch[0,12],roll[-6,6] degrees, all combinations. Recursive interval bound: triangle-surface nearest distance minus maximum rotational displacement. Margin1.0mm; motor and projector convex hulls used as conservative envelopes. Conservative radius includes48mm pivot separation allowance. Does not alone prove initial containment, dimensional tolerances, deformation, or parts omitted from pairs. Existing scene includes no complete v9 support frame.',rows},null,2));
 }
}
