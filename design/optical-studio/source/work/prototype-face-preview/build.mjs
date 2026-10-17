import fs from 'node:fs';
import * as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {STLLoader} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/loaders/STLLoader.js';
import {mergeGeometries} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/utils/BufferGeometryUtils.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {projectionPath} from '../../outputs/danya-simulator/src/optics.js';
const root='outputs/danya-simulator/assets/projection-prototype',out=root+'/face-preview';fs.mkdirSync(out,{recursive:true});
const state=JSON.parse(fs.readFileSync(root+'/layout.json')),manifest=JSON.parse(fs.readFileSync(root+'/parts.json'));
function load(name){let b=fs.readFileSync(root+'/assembly/'+name+'.stl');let g=new STLLoader().parse(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));for(const n of Object.keys(g.attributes))if(n!=='position')g.deleteAttribute(n);return g;}
const face=load('face'),pos=face.attributes.position,uv=new Float32Array(pos.count*2),lit=new Float32Array(pos.count),faceBVH=new MeshBVH(face);
const parts=manifest.parts.filter(p=>!p.template&&!['face','mirror-glass','foldMirror-glass'].includes(p.name));const merged=mergeGeometries(parts.map(p=>load(p.name))),bvh=new MeshBVH(merged);
const cache=new Map();let valid=0,unblocked=0;
function blocked(a,b,tree){const d=b.clone().sub(a),len=d.length();return !!tree.raycastFirst(new T.Ray(a,d.divideScalar(len)),T.DoubleSide,.1,len-.1);}
for(let i=0;i<pos.count;i++){const q=new T.Vector3().fromBufferAttribute(pos,i),key=q.toArray().join(',');let record=cache.get(key);if(!record){const p=projectionPath(state,q);let ok=!!p;if(p){valid++;for(let j=1;j<p.legs.length&&ok;j++){const a=p.legs[j-1],b=p.legs[j];if(blocked(a,b,bvh)||j<p.legs.length-1&&blocked(a,b,faceBVH))ok=false;}if(ok){const a=p.legs.at(-2),d=q.clone().sub(a),len=d.length(),ray=new T.Ray(a,d.divideScalar(len));const hits=faceBVH.raycast(ray,T.DoubleSide,.1,len-.1).sort((a,b)=>a.distance-b.distance),unique=hits.filter((h,j)=>j===0||h.distance-hits[j-1].distance>.1);ok=unique.length===0||(unique.length===1&&unique[0].face.normal.dot(ray.direction)<0);}}record=[p?.u??0,p?.v??0,ok?1:0];cache.set(key,record);if(ok)unblocked++;}uv[i*2]=record[0];uv[i*2+1]=record[1];lit[i]=record[2];}
fs.writeFileSync(out+'/uv.bin',Buffer.from(uv.buffer));fs.writeFileSync(out+'/lit.bin',Buffer.from(lit.buffer));fs.writeFileSync(out+'/metadata.json',JSON.stringify({vertices:pos.count,uniqueVertices:cache.size,validPaths:valid,unblockedVertices:unblocked,blockerParts:parts.length,layout:state,scope:'Point-source vertex sampling. Last continuous face layer treated as transmitting screen. Not measured brightness, diffusion, focus or full-face coverage certification.'},null,2));console.log({vertices:pos.count,unique:cache.size,valid,unblocked});
