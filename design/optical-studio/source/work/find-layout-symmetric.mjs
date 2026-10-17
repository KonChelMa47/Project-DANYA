import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {Vector3,Quaternion,Matrix4,BufferGeometry,BufferAttribute,Ray,DoubleSide} from 'three';
import {MeshBVH} from 'three-mesh-bvh';
import {defaultState,degrees,centerLocal,projectionPath,visiblePath,basis} from '../src/optics.js';
const root=fileURLToPath(new URL('../',import.meta.url));
function geometry(name){let g=new BufferGeometry();for(const [key,type,size]of [['positions',Float32Array,3],['indices',Uint32Array,1]]){let b=fs.readFileSync(root+'assets/'+name+'.'+key+'.bin'),a=new type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));if(key==='indices')g.setIndex(new BufferAttribute(a,size));else g.setAttribute('position',new BufferAttribute(a,size));}return g;}
const fg=geometry('face'),hg=geometry('head'),face=new MeshBVH(fg),head=new MeshBVH(hg),V=(x,y,z)=>new Vector3(x,y,z);
let seed=751823;function rand(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
const samples=[]; // Equal projected-area frontal samples, including nose/lips/edge regions.
for(let z=44.2;z<=221;z+=5)for(let y=-64;y<=65;y+=5){const hit=face.raycastFirst(new Ray(V(400,y,z),V(-1,0,0)),DoubleSide,0,500);if(hit)samples.push(hit.point);}
console.log('samples',samples.length);
// parameters: mirror x,z, normal downward angle; lens x,z; aim point along mirror local vertical; roll
const bounds=[[-75,20],[85,210],[15,75],[-70,100],[-180,55],[-60,60],[-18,18]];
function make(a){const s=defaultState();s.mirror.width=Number(process.env.SIM_MIRROR_WIDTH||110);s.mirror.height=Number(process.env.SIM_MIRROR_HEIGHT||115);const n=V(Math.cos(a[2]*Math.PI/180),0,-Math.sin(a[2]*Math.PI/180)),r=V(0,1,0),u=new Vector3().crossVectors(n,r);s.mirror.position=[a[0],0,a[1]];s.mirror.rotation=degrees(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(r,u,n)));s.projector.position=[a[3],0,a[4]];const target=V(...s.mirror.position).addScaledVector(u,a[5]),d=target.sub(V(...s.projector.position)).normalize(),q=new Quaternion().setFromUnitVectors(centerLocal(s.optics),d);q.premultiply(new Quaternion().setFromAxisAngle(d,a[6]*Math.PI/180));s.projector.rotation=degrees(q);s.aimRoll=a[6];return s;}
function score(a,all=false){const s=make(a);let good=0,geom=0,margin=0,low=0,high=0;for(let i=0;i<samples.length;i++){const p=samples[i],path=projectionPath(s,p);if(!path)continue;geom++;if(!visiblePath(s,p,head,face,path))continue;good++;margin+=Math.min(path.u,1-path.u,path.v,1-path.v);if(p.z<85)low++;if(p.z>190)high++;}return {value:good/samples.length+margin/samples.length*.0001,good,geom,total:samples.length,state:s,params:a};}
let pop=Array.from({length:70},(_,i)=>score(i===0?JSON.parse(fs.readFileSync(root+'search-result.json')).params:bounds.map(([a,b])=>a+rand()*(b-a))));let best=pop.reduce((a,b)=>a.value>b.value?a:b);console.log('initial',best.good,best.params);
for(let gen=0;gen<60;gen++){for(let i=0;i<pop.length;i++){let ids=[];while(ids.length<3){let k=Math.floor(rand()*pop.length);if(k!==i&&!ids.includes(k))ids.push(k);}const force=Math.floor(rand()*bounds.length),a=pop[i].params.map((v,j)=>{if(rand()>.8&&j!==force)return v;let x=pop[ids[0]].params[j]+.7*(pop[ids[1]].params[j]-pop[ids[2]].params[j]);return Math.max(bounds[j][0],Math.min(bounds[j][1],x));});const candidate=score(a);if(candidate.value>pop[i].value)pop[i]=candidate;if(candidate.value>best.value){best=candidate;fs.writeFileSync(root+'candidate-layout.json',JSON.stringify(best.state,null,2));fs.writeFileSync(root+'search-progress.json',JSON.stringify({generation:gen,...best},null,2));}}
 if(gen%3===0)console.log('generation',gen,'coverage',best.good+'/'+samples.length,'params',best.params.map(x=>+x.toFixed(3)));}
console.log(JSON.stringify(best));fs.writeFileSync(root+'search-result.json',JSON.stringify(best,null,2));
