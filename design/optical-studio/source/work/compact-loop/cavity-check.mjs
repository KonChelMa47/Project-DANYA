import fs from 'node:fs';
import {BufferGeometry,BufferAttribute,Matrix4} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const geo=(p,i)=>new BufferGeometry().setAttribute('position',new BufferAttribute(read(p,Float32Array),3)).setIndex(new BufferAttribute(read(i,Uint32Array),1));
const h=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','work/adult-kit/head-outer-original.indices.bin')),dir='work/compact-loop/cavity-v2/',results={};
for(const file of fs.readdirSync(dir).filter(f=>f.endsWith('.positions.bin')&&f!=='beam3.positions.bin'&&f!=='head.positions.bin')){const name=file.replace('.positions.bin',''),g=geo(dir+file,dir+name+'.indices.bin');g.boundsTree=new MeshBVH(g);const intersects=h.intersectsGeometry(g,new Matrix4()),a={};h.closestPointToGeometry(g,new Matrix4(),a,{},1,1);results[name]={intersectsHeadExterior:intersects,atLeast1MM:!intersects&&!(a.distance<1),nearestExaminedDistance:a.distance};console.log(name,results[name]);}
fs.writeFileSync(dir+'cutters-exterior.json',JSON.stringify(results,null,2));
