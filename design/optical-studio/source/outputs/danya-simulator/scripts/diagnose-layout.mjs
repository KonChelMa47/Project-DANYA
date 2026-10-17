import fs from 'node:fs';import {fileURLToPath} from 'node:url';
import {Vector3,BufferGeometry,BufferAttribute,Ray,DoubleSide} from 'three';import {MeshBVH} from 'three-mesh-bvh';
import {defaultState,projectionPath,visiblePath,blocked,vec,basis} from '../src/optics.js';
const root=fileURLToPath(new URL('../',import.meta.url)),V=(x,y,z)=>new Vector3(x,y,z);
function geometry(name){const g=new BufferGeometry();for(const [k,T,n]of [['positions',Float32Array,3],['indices',Uint32Array,1]]){const b=fs.readFileSync(root+'assets/'+name+'.'+k+'.bin'),a=new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));if(k==='indices')g.setIndex(new BufferAttribute(a,n));else g.setAttribute('position',new BufferAttribute(a,n));}return g;}
const face=new MeshBVH(geometry('face')),head=new MeshBVH(geometry('head'));
const state=process.argv[2]==='initial'?defaultState():JSON.parse(fs.readFileSync(root+(process.argv[2]||'candidate-layout.json'))),big=structuredClone(state);big.mirror.width=big.mirror.height=2000;
let counts={lit:0,mirrorClip:0,imageClip:0,firstLeg:0,headSecond:0,faceSelf:0},regions={},miss=[],lengths=[];
const step=Number(process.argv[3]||2);
for(let z=43.33;z<221;z+=step)for(let y=-64.66;y<65.23;y+=step){const ray=new Ray(V(400,y,z),V(-1,0,0)),hit=face.raycastFirst(ray,DoubleSide,0,500);if(!hit)continue;const hh=head.raycastFirst(ray,DoubleSide,0,500);if(hh&&hh.distance<hit.distance-.1)continue;const p=hit.point,path=projectionPath(state,p);let kind;
 if(!path)kind=projectionPath(big,p)?'mirrorClip':'imageClip';else if(blocked(head,vec(state.projector.position),path.hit)||blocked(face,vec(state.projector.position),path.hit))kind='firstLeg';else if(blocked(head,path.hit,p))kind='headSecond';else if(!visiblePath(state,p,head,face,path))kind='faceSelf';else {kind='lit';lengths.push(path.distance);}
 counts[kind]++;const region=z<85?'chin':z<125?'mouth':z<165?'nose-cheeks':z<195?'eyes':'forehead';regions[region]??={lit:0,total:0};regions[region].total++;if(kind==='lit')regions[region].lit++;else miss.push({p:p.toArray(),kind});
}
const total=Object.values(counts).reduce((a,b)=>a+b,0),out={state,counts,total,coverage:counts.lit/total,regions,pathMin:Math.min(...lengths),pathMax:Math.max(...lengths),step,miss};
// Test the finite mirror surface against shell material by edge/row segment intersections.
const m=basis(state.mirror);let intersectingRows=0;
for(let i=0;i<=50;i++){const a=m.p.clone().addScaledVector(m.u,(i/50-.5)*state.mirror.height).addScaledVector(m.r,-state.mirror.width/2),b=a.clone().addScaledVector(m.r,state.mirror.width);if(blocked(head,a,b)||blocked(face,a,b))intersectingRows++;}
out.mirrorIntersectingRows=intersectingRows;fs.writeFileSync(root+'diagnosis-'+(process.argv[2]||'candidate').replace('.json','')+'.json',JSON.stringify(out,null,2));console.log(JSON.stringify({...out,miss:miss.filter((_,i)=>i%Math.max(1,Math.floor(miss.length/10))===0)},null,2));
