import fs from 'node:fs';
import {Vector3,Quaternion,Matrix4,BufferGeometry,BufferAttribute,Ray,DoubleSide,PlaneGeometry,BoxGeometry} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {defaultState,degrees,centerLocal,projectionPath,visiblePath,basis} from '../../outputs/danya-simulator/src/optics.js';
const root=new URL('../../outputs/danya-simulator/',import.meta.url),out=new URL('./',import.meta.url),V=(x,y,z)=>new Vector3(x,y,z);
function geom(name,k=1,sy=1){let g=new BufferGeometry();for(const [key,T,n]of [['positions',Float32Array,3],['indices',Uint32Array,1]]){let b=fs.readFileSync(new URL('assets/'+name+'.'+key+'.bin',root)),a=new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));if(key==='indices')g.setIndex(new BufferAttribute(a,n));else {if(name==='head')for(let i=0;i<a.length;i+=3){const t=Math.min(1,Math.max(0,-a[i]/25)),w=t*t*(3-2*t);a[i]*=1+(k-1)*w;a[i+1]*=1+(sy-1)*w;}g.setAttribute('position',new BufferAttribute(a,n));}}return g;}
const face=new MeshBVH(geom('face')),original=new MeshBVH(geom('head')),samples=[];
for(let z=44.2;z<221;z+=5)for(let y=-64;y<65.3;y+=5){let ray=new Ray(V(400,y,z),V(-1,0,0)),hit=face.raycastFirst(ray,DoubleSide,0,500),hh=original.raycastFirst(ray,DoubleSide,0,500);if(hit&&(!hh||hh.distance>=hit.distance-.1))samples.push(hit.point);}
function make(mx,mz,px,pz,tz){const s=defaultState();s.projector.position=[px,0,pz];s.mirror.position=[mx,0,mz];const m=V(mx,0,mz),p=V(px,0,pz),incoming=m.clone().sub(p).normalize(),outgoing=V(55,0,tz).sub(m).normalize(),n=outgoing.clone().sub(incoming).normalize(),r=V(0,1,0),u=new Vector3().crossVectors(n,r);s.mirror.rotation=degrees(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(r,u,n)));s.projector.rotation=degrees(new Quaternion().setFromUnitVectors(centerLocal(s.optics),incoming));s.mirror.width=160;s.mirror.height=230;return s;}
let best=null;const plane=new PlaneGeometry(1,1);
for(const [k,sy]of [[1,1],[1.5,1.1],[1.8,1.15],[2.1,1.2],[2.5,1.3]]){const head=new MeshBVH(geom('head',k,sy));let local=null;
 for(const mx of [-35,-55,-75,-95])for(const mz of [125,145,165])for(const px of [-50,-75,-100])for(const pz of [-80,-140,-200,-260])for(const tz of [125,145,160]){
  const s=make(mx,mz,px,pz,tz);let good=0,minUV=1,minM=1e9,points=[];for(const q of samples){const path=projectionPath(s,q);if(path&&visiblePath(s,q,head,face,path)){good++;minUV=Math.min(minUV,path.u,1-path.u,path.v,1-path.v);points.push(path.hit.toArray());}}
  const score=good/samples.length+minUV*.0001;if(!local||score>local.score){local={score,good,total:samples.length,k,sy,state:s,minUV};}
  if(!best||score>best.score){best={score,good,total:samples.length,k,sy,state:s,minUV};fs.writeFileSync(new URL('best.json',out),JSON.stringify(best,null,2));}
 }
 console.log(JSON.stringify(local));}
console.log('BEST',JSON.stringify(best));
