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
const fullHead=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','outputs/danya-simulator/assets/head.indices.bin')),fullFace=new MeshBVH(geo('work/torso-flat/face-radial-0.8.positions.bin','work/torso-flat/face-radial-0.8.indices.bin'));
const points=[...new Set(fg.index.array)].map(i=>new Vector3().fromBufferAttribute(fg.attributes.position,i));
for(let i=0;i<fg.index.count;i+=3){let p=new Vector3();for(let j=0;j<3;j++)p.add(new Vector3().fromBufferAttribute(fg.attributes.position,fg.index.array[i+j]));points.push(p.divideScalar(3));}
const {buildProjectionMask,maskAt}=await import('../../outputs/danya-simulator/src/projection-mask.js');const hnew=new MeshBVH(geo('work/torso-flat/shell/bevel-0.6.positions.bin','work/torso-flat/shell/bevel-0.6.indices.bin'));const bounds={min:[1,1],max:[0,0]},paths=[];for(const q of points){const p=projectionPath(s,q);paths.push(p);if(p)for(let i=0;i<2;i++){const a=i?p.v:p.u;bounds.min[i]=Math.min(bounds.min[i],a);bounds.max[i]=Math.max(bounds.max[i],a);}}let last=-1;const mask=await buildProjectionMask(s,hnew,fullFace,kit,bounds,async p=>{const pc=Math.floor(p*10);if(pc!==last){last=pc;console.log('mask',pc*10);}return true;});let missed=0;for(const p of paths)if(!p||!maskAt(mask,p.u,p.v))missed++;fs.writeFileSync('work/torso-flat/nominal-mask.bin',mask);fs.writeFileSync('work/torso-flat/nominal-mask.json',JSON.stringify({scope:'Official complete body, rebuilt head, radial experimental full face. Nine chief samples per native pixel; finite pupil not covered.',bounds,total:points.length,maskedTargets:missed,enabledPixels:mask.reduce((a,v)=>a+(v===255),0)},null,2));console.log({total:points.length,missed});
