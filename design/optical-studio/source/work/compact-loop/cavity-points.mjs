import fs from 'node:fs';
import {Vector3,Matrix4,BufferGeometry,BufferAttribute} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {cadEngine,kit} from '../adult-kit/load-model.mjs';
import {projectionPath,basis} from '../../outputs/danya-simulator/src/optics.js';
import {mirrorSolidGeometry} from '../../outputs/danya-simulator/src/mirror-geometry.js';
const source=process.argv[2];if(!source)throw Error('Pass validated candidate state JSON');
const s=JSON.parse(fs.readFileSync(source));const out='work/compact-loop/cavity-v2/';
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const v=read('work/adult-kit/original-face.positions.bin',Float32Array),idx=read('work/adult-kit/original-face-outer.indices.bin',Uint32Array),sets=Array.from({length:4},()=>new Map());
for(const i of new Set(idx)){const q=new Vector3().fromArray(v,i*3),p=projectionPath(s,q);if(!p)throw Error('Missing face vertex');for(let j=0;j<4;j++)for(const a of [p.legs[j],p.legs[j+1]])sets[j].set(a.toArray().map(x=>x.toFixed(5)).join(','),a.toArray());}
function save(name,points,pad){fs.writeFileSync(out+name+'.points.json',JSON.stringify({points,pad}));}
sets.forEach((set,i)=>save('beam'+i,[...set.values()],.3));
const matrix=p=>{const b=basis(p);return new Matrix4().makeBasis(b.r,b.u,b.f).setPosition(b.p);};
for(const[name,g]of [['official-body',kit],['tripo-body',cadEngine],...['preMirror','foldMirror','mirror'].map(k=>[k,mirrorSolidGeometry(s[k]).scale(s[k].width,s[k].height,1).translate(0,0,-.85)])]){const m=matrix(s[name]||s.projector),points=[];for(let i=0;i<g.attributes.position.count;i++)points.push(new Vector3().fromBufferAttribute(g.attributes.position,i).applyMatrix4(m).toArray());save(name,points,.3);}
fs.writeFileSync(out+'state.json',JSON.stringify(s,null,2));console.log('Saved bounded cutter point clouds; no shell modified.');
