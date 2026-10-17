import fs from 'node:fs';
import {Vector3,Ray,DoubleSide,BufferGeometry,BufferAttribute,Matrix4,Box3} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {cadEngine,kit} from '../adult-kit/load-model.mjs';
import {basis,projectionPath,visiblePath,rayForUV} from '../../outputs/danya-simulator/src/optics.js';
import {createStrayTracer} from '../../outputs/danya-simulator/src/stray-light.js';
const s=JSON.parse(fs.readFileSync('work/torso-flat/layout.json'));
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const geo=(p,i)=>new BufferGeometry().setAttribute('position',new BufferAttribute(read(p,Float32Array),3)).setIndex(new BufferAttribute(read(i,Uint32Array),1));
const fg=geo('work/adult-kit/original-face.positions.bin','work/adult-kit/original-face-outer.indices.bin'),f=new MeshBVH(fg),h=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','work/adult-kit/head-outer-original.indices.bin'));
const fullHead=new MeshBVH(geo('work/torso-flat/shell/bevel-0.6.positions.bin','work/torso-flat/shell/bevel-0.6.indices.bin')),fullFace=new MeshBVH(geo('work/torso-flat/face-local-v2.positions.bin','work/torso-flat/face-local-v2.indices.bin'));
const points=[...new Set(fg.index.array)].map(i=>new Vector3().fromBufferAttribute(fg.attributes.position,i));
for(let i=0;i<fg.index.count;i+=3){let p=new Vector3();for(let j=0;j<3;j++)p.add(new Vector3().fromBufferAttribute(fg.attributes.position,fg.index.array[i+j]));points.push(p.divideScalar(3));}
let report=[];for(const bevel of[0.6,1]){const hb=new MeshBVH(geo(`work/torso-flat/shell/bevel-${bevel.toFixed(1)}.positions.bin`,`work/torso-flat/shell/bevel-${bevel.toFixed(1)}.indices.bin`)),trace=createStrayTracer(s,hb,fullFace,kit);let n=0,kinds={},fail=[];for(let i=0;i<points.length;i+=250){const q=points[i],p=projectionPath(s,q);if(!p)continue;const out=trace(p.u,p.v);n++;kinds[out.kind]=(kinds[out.kind]||0)+1;if(out.kind!=='face')fail.push({target:q.toArray(),kind:out.kind,route:out.route,points:out.points.map(v=>v.toArray())});}report.push({bevel,n,kinds,fail});console.log({bevel,n,kinds});}fs.writeFileSync('work/torso-flat/head-seam-sweep.json',JSON.stringify(report,null,2));