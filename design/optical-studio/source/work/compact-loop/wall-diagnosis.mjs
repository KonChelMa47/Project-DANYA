import fs from 'node:fs';
import {Vector3,Ray,DoubleSide,BufferGeometry,BufferAttribute,Matrix4,Box3} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {cadEngine,kit} from '../adult-kit/load-model.mjs';
import {basis,projectionPath,visiblePath,rayForUV,blocked} from '../../outputs/danya-simulator/src/optics.js';
import {createStrayTracer} from '../../outputs/danya-simulator/src/stray-light.js';
const s=JSON.parse(fs.readFileSync('work/compact-loop/margin-candidate.json'));
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const geo=(p,i)=>new BufferGeometry().setAttribute('position',new BufferAttribute(read(p,Float32Array),3)).setIndex(new BufferAttribute(read(i,Uint32Array),1));
const fg=geo('work/adult-kit/original-face.positions.bin','work/adult-kit/original-face-outer.indices.bin'),f=new MeshBVH(fg),h=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','work/adult-kit/head-outer-original.indices.bin'));
const fullHead=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','outputs/danya-simulator/assets/head.indices.bin')),fullFace=new MeshBVH(geo('work/adult-kit/original-face.positions.bin','outputs/danya-simulator/assets/face.indices.bin'));
const points=[...new Set(fg.index.array)].map(i=>new Vector3().fromBufferAttribute(fg.attributes.position,i));
for(let i=0;i<fg.index.count;i+=3){let p=new Vector3();for(let j=0;j<3;j++)p.add(new Vector3().fromBufferAttribute(fg.attributes.position,fg.index.array[i+j]));points.push(p.divideScalar(3));}
const result=[];for(let i=0;i<points.length;i+=500){const q=points[i],path=projectionPath(s,q);if(!path)continue;const ray=new Ray(path.hit,q.clone().sub(path.hit).normalize()),len=path.hit.distanceTo(q),hits=fullHead.raycast(ray,DoubleSide,.08,len-.08).sort((a,b)=>a.distance-b.distance);if(!hits.length)continue;const occupancy=fullHead.raycast(new Ray(q,new Vector3(.932,.271,.241).normalize()),DoubleSide,.0001,1000).sort((a,b)=>a.distance-b.distance);const distinct=occupancy.filter((a,i)=>!i||a.distance-occupancy[i-1].distance>1e-4);const near={};h.closestPointToPoint(q,near);result.push({q:q.toArray(),insideHead:distinct.length%2===1,headOuterDistance:near.distance,hits:hits.map(a=>({point:a.point.toArray(),distanceToTarget:len-a.distance,normal:a.face.normal.toArray()}))});}fs.writeFileSync('work/compact-loop/wall-diagnosis.json',JSON.stringify(result,null,2));console.log({sampleBlocked:result.length,insideHead:result.filter(r=>r.insideHead).length,first:result.slice(0,3)});
