import fs from 'node:fs';
import {Vector3,BufferGeometry,BufferAttribute,Float32BufferAttribute,Matrix4} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {ConvexHull} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/math/ConvexHull.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {basis} from '../../outputs/danya-simulator/src/optics.js';
const s=JSON.parse(fs.readFileSync('work/torso-flat/torso-v3-layout.json')),points=[];
for(const p of[s.foldMirror,s.mirror]){const b=basis(p);for(const[x,y]of p.outline||[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])points.push(b.p.clone().addScaledVector(b.r,x*p.width).addScaledVector(b.u,y*p.height));}
function convex(points){const hull=new ConvexHull().setFromPoints(points),flat=[];for(const f of hull.faces){let e=f.edge;do{flat.push(...e.head().point.toArray());e=e.next;}while(e!==f.edge);}return{hull,bvh:new MeshBVH(new BufferGeometry().setAttribute('position',new Float32BufferAttribute(flat,3)))};}
const b=basis(s.projector),mat=new Matrix4().makeBasis(b.r,b.u,b.f).setPosition(b.p),targets={optical:convex(points),body:convex(JSON.parse(fs.readFileSync('work/torso-flat/projector-hull.json')).vertices.map(p=>new Vector3(...p).applyMatrix4(mat)))};
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
for(const[name,p]of[['head','work/torso-flat/shell/bevel-0.6'],['face','work/torso-flat/face-local-v2']]){const g=new BufferGeometry().setAttribute('position',new BufferAttribute(read(p+'.positions.bin',Float32Array),3)).setIndex(new BufferAttribute(read(p+'.indices.bin',Uint32Array),1));targets[name]={bvh:new MeshBVH(g)};}
const rows=JSON.parse(fs.readFileSync('work/torso-flat/projector-edge-clamp-lines.json')).rows;
for(const row of rows){row.clearanceLowerBoundsMM={};const a=new Vector3(...row.points[0]),b=new Vector3(...row.points[1]),n=Math.ceil(a.distanceTo(b)/.5);for(const name of ['optical']){let d=Infinity;for(let i=0;i<=n;i++){const p=a.clone().lerp(b,i/n),hit={};targets[name].bvh.closestPointToPoint(p,hit);d=Math.min(d,targets[name].hull.containsPoint(p)?-hit.distance:hit.distance);}row.clearanceLowerBoundsMM[name]=d-4.25-.25001;}}
fs.writeFileSync('work/torso-flat/projector-edge-clamp-optical.json',JSON.stringify({scope:'Radius4.25 rails, 0.5mm centerline samples and 0.25001 guard vs fold-main convex solid; intended base contact excludes body test. First leg requires separate validation.',rows},null,2));console.log(rows);
