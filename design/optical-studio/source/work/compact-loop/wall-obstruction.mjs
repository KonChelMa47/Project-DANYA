import fs from 'node:fs';
import {Vector3,Ray,DoubleSide,BufferGeometry,BufferAttribute,Matrix4,Box3} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {cadEngine,kit} from '../adult-kit/load-model.mjs';
import {basis,projectionPath,visiblePath,rayForUV} from '../../outputs/danya-simulator/src/optics.js';
import {createStrayTracer} from '../../outputs/danya-simulator/src/stray-light.js';
const s=JSON.parse(fs.readFileSync('work/compact-loop/dual-routing-candidate.json'));
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const geo=(p,i)=>new BufferGeometry().setAttribute('position',new BufferAttribute(read(p,Float32Array),3)).setIndex(new BufferAttribute(read(i,Uint32Array),1));
const fg=geo('work/adult-kit/original-face.positions.bin','work/adult-kit/original-face-outer.indices.bin'),f=new MeshBVH(fg),h=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','work/adult-kit/head-outer-original.indices.bin'));
const fullHead=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','outputs/danya-simulator/assets/head.indices.bin')),fullFace=new MeshBVH(geo('work/adult-kit/original-face.positions.bin','outputs/danya-simulator/assets/face.indices.bin'));
const points=[...new Set(fg.index.array)].map(i=>new Vector3().fromBufferAttribute(fg.attributes.position,i));
for(let i=0;i<fg.index.count;i+=3){let p=new Vector3();for(let j=0;j<3;j++)p.add(new Vector3().fromBufferAttribute(fg.attributes.position,fg.index.array[i+j]));points.push(p.divideScalar(3));}
let counts={total:0,outer:0,headWall:0,faceWall:0,both:0};for(let i=0;i<points.length;i+=10){const q=points[i],path=projectionPath(s,q);counts.total++;if(visiblePath(s,q,h,f,path))counts.outer++;if(visiblePath(s,q,fullHead,f,path))counts.headWall++;if(visiblePath(s,q,h,fullFace,path))counts.faceWall++;if(visiblePath(s,q,fullHead,fullFace,path))counts.both++;}console.log(counts);fs.writeFileSync('work/compact-loop/wall-obstruction.json',JSON.stringify(counts,null,2));
