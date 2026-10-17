import fs from 'node:fs';import {Vector3,Matrix4,BufferGeometry,BufferAttribute,Ray,DoubleSide} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';import {projectionPath,blocked,vec,quaternion} from '../../outputs/danya-simulator/src/optics.js';import {mirrorSolidGeometry} from '../../outputs/danya-simulator/src/mirror-geometry.js';
const dir=new URL('./',import.meta.url),d=JSON.parse(fs.readFileSync(new URL('design.json',dir))),s=d.state,V=(x,y,z)=>new Vector3(x,y,z);
function arr(name,key,T){const b=fs.readFileSync(new URL(name+'.'+key+'.bin',dir));return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
const fv=arr('face','positions',Float32Array),fi=arr('face','indices',Uint32Array),hv=arr('head','positions',Float32Array),hi=arr('head','indices',Uint32Array);
function geo(v,i){const g=new BufferGeometry();g.setAttribute('position',new BufferAttribute(v,3));g.setIndex(new BufferAttribute(i.slice(),1));return g;}
const face=new MeshBVH(geo(fv,fi)),paths=[];for(let z=43;z<222;z+=3)for(let y=-66;y<66;y+=3){const hit=face.raycastFirst(new Ray(V(400,y,z),V(-1,0,0)),DoubleSide,0,700);if(hit){const path=projectionPath(s,hit.point);if(path)paths.push(path);}}
const mg=mirrorSolidGeometry(s.mirror),mm=new Matrix4().compose(vec(s.mirror.position),quaternion(s.mirror.rotation),V(1,1,1)).multiply(new Matrix4().makeTranslation(0,0,-.85)).multiply(new Matrix4().makeScale(s.mirror.width,s.mirror.height,1));
const smooth=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
let best;
for(const amount of [0,15,25,35,45,55,70]){const v=hv.slice();for(let i=0;i<v.length;i+=3){const weight=smooth((-v[i]-5)/35)*(1-smooth((v[i+2]-60)/80));v[i]-=amount*weight;}
const head=new MeshBVH(geo(v,hi));let blockedRays=0;for(const p of paths)if(blocked(head,vec(s.projector.position),p.hit))blockedRays++;const collision=head.intersectsGeometry(mg,mm);console.log({amount,blockedRays,total:paths.length,collision});if(!blockedRays&&!collision){best={amount,v};break;}}
if(best){fs.writeFileSync(new URL('head.positions.bin',dir),Buffer.from(best.v.buffer));d.neckRearShiftMM=best.amount;d.neckFade={xStart:-5,xFull:-40,zFull:60,zEnd:140};fs.writeFileSync(new URL('design.json',dir),JSON.stringify(d,null,2));}else process.exitCode=2;
