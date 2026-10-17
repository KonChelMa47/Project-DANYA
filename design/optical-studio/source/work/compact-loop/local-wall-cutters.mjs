import fs from 'node:fs';
import {Vector3,Ray,DoubleSide,BufferGeometry,BufferAttribute} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {projectionPath} from '../../outputs/danya-simulator/src/optics.js';
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const geo=(p,i)=>new BufferGeometry().setAttribute('position',new BufferAttribute(read(p,Float32Array),3)).setIndex(new BufferAttribute(read(i,Uint32Array),1));
const s=JSON.parse(fs.readFileSync('work/compact-loop/margin-candidate.json')),head=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','outputs/danya-simulator/assets/head.indices.bin')),fg=geo('work/adult-kit/original-face.positions.bin','work/adult-kit/original-face-outer.indices.bin'),patches=new Map(),points=[...new Set(fg.index.array)].map(i=>new Vector3().fromBufferAttribute(fg.attributes.position,i));
for(let i=0;i<fg.index.count;i+=3){const q=new Vector3();for(let j=0;j<3;j++)q.add(new Vector3().fromBufferAttribute(fg.attributes.position,fg.index.array[i+j]));points.push(q.divideScalar(3));}
let blocked=0;const bounds=[new Vector3(Infinity,Infinity,Infinity),new Vector3(-Infinity,-Infinity,-Infinity)];
for(const q of points){const p=projectionPath(s,q);if(!p)throw Error('No chief ray');const d=q.clone().sub(p.hit),len=d.length(),ray=new Ray(p.hit,d.divideScalar(len)),hits=head.raycast(ray,DoubleSide,.08,len-.08).sort((a,b)=>a.distance-b.distance);if(!hits.length)continue;blocked++;const key=Math.floor(q.y/10)+'_'+Math.floor(q.z/10),a=ray.at(Math.max(.1,hits[0].distance-.3),new Vector3()),b=ray.at(Math.min(len-.1,hits.at(-1).distance+.3),new Vector3());for(const pt of[a,b]){bounds[0].min(pt);bounds[1].max(pt);}(patches.get(key)||patches.set(key,[]).get(key)).push(a.toArray(),b.toArray());}
const out='work/compact-loop/cavity-v2/';for(const[k,points]of patches)fs.writeFileSync(out+'wall_'+k+'.points.json',JSON.stringify({points,pad:.15}));
console.log({blocked,patches:patches.size,bounds:bounds.map(v=>v.toArray())});
