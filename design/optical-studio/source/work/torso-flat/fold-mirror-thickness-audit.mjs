import fs from 'node:fs';
import{Vector3,BufferGeometry,Float32BufferAttribute,Matrix4}from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import{ConvexHull}from '../../outputs/danya-simulator/node_modules/three/examples/jsm/math/ConvexHull.js';
import{MeshBVH}from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import{basis}from '../../outputs/danya-simulator/src/optics.js';
const s=JSON.parse(fs.readFileSync('work/torso-flat/torso-v3-layout.json')),f=basis(s.foldMirror),p=basis(s.projector),main=basis(s.mirror),thickness=2;
const corners=(part,b,z=0)=>(part.outline||[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).map(([x,y])=>b.p.clone().addScaledVector(b.r,x*part.width).addScaledVector(b.u,y*part.height).addScaledVector(b.f,z));
const pupil=[];for(const x of[-4,4])for(const y of[-4,4])pupil.push(p.p.clone().addScaledVector(p.r,x).addScaledVector(p.u,y));
const distance=v=>v.clone().sub(f.p).dot(f.f),front={pupilMin:Math.min(...pupil.map(distance)),mainMirrorMin:Math.min(...corners(s.mirror,main).map(distance))};
function hull(ps){const h=new ConvexHull().setFromPoints(ps),a=[];for(const face of h.faces){let e=face.edge;do{a.push(...e.head().point.toArray());e=e.next;}while(e!==face.edge);}return{h,g:new BufferGeometry().setAttribute('position',new Float32BufferAttribute(a,3))};}
const m=new Matrix4().makeBasis(p.r,p.u,p.f).setPosition(p.p),body=hull(JSON.parse(fs.readFileSync('work/torso-flat/projector-hull.json')).vertices.map(v=>new Vector3(...v).applyMatrix4(m))),mirrorPoints=[...corners(s.foldMirror,f),...corners(s.foldMirror,f,-thickness)],mirror=hull(mirrorPoints),hit={};new MeshBVH(body.g).closestPointToGeometry(mirror.g,new Matrix4(),hit);
const report={scope:'Provisional full rectangular 2mm front-surface small mirror. Halfspace proof covers assumed8mm pupil and full main mirror hull; unrelated stray rays not covered. Backing is rear disk localZ[-4,-2.2] and rear stem[-8,-4].',frontHalfspaceMM:front,backingMaxLocalZMM:-2.2,firstAndSecondLegStayInFront:front.pupilMin>0&&front.mainMirrorMin>0,bodyToMirrorSurfaceMM:hit.distance,mirrorVertexInsideBody:mirrorPoints.some(v=>body.h.containsPoint(v)),mirrorWorldBounds:{min:[0,1,2].map(i=>Math.min(...mirrorPoints.map(v=>v.getComponent(i)))),max:[0,1,2].map(i=>Math.max(...mirrorPoints.map(v=>v.getComponent(i))))}};
fs.writeFileSync('work/torso-flat/fold-mirror-thickness-audit.json',JSON.stringify(report,null,2));console.log(report);
