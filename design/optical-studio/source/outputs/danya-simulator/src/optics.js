import {apertureBladeBlocks} from './aperture-blade.js';
import {MeshBVH} from 'three-mesh-bvh';
const cadEngineBVHs={};
export function setCADEngineGeometry(geometry,key='ti-engine-rd'){cadEngineBVHs[key]=geometry.boundsTree||new MeshBVH(geometry);geometry.boundsTree=cadEngineBVHs[key];}
import {curvedReflection} from './curved-mirror.js';
import {Vector3, Quaternion, Euler, Matrix4, Ray, DoubleSide, Box3} from 'three';

export const SPEC = Object.freeze({width:1280,height:720,throwRatio:1.2,offset:100,lumens:220,focusMin:127,focusMax:1270});
export const vec = a => new Vector3(...a);
export const quaternion = degrees => new Quaternion().setFromEuler(new Euler(...degrees.map(d=>d*Math.PI/180),'XYZ'));
export const degrees = q => new Euler().setFromQuaternion(q,'XYZ').toArray().slice(0,3).map(x=>x*180/Math.PI);
export function reflect(direction, normal) { return direction.clone().addScaledVector(normal,-2*direction.dot(normal)); }
export function reflectPoint(point,center,normal) {return point.clone().addScaledVector(normal,-2*point.clone().sub(center).dot(normal));}
// Convex, counterclockwise normalized outline. Binary-search its triangle fan.
export function mirrorContains(mirror,x,y) {
  if(Math.abs(x)>mirror.width/2 || Math.abs(y)>mirror.height/2)return false;
  const poly=mirror.outline;if(!poly)return true;
  const p=[x/mirror.width,y/mirror.height],a=poly[0];
  const cross=(u,v,w)=>(v[0]-u[0])*(w[1]-u[1])-(v[1]-u[1])*(w[0]-u[0]);
  if(cross(a,poly[1],p)<-1e-10||cross(a,poly.at(-1),p)>1e-10)return false;
  let lo=1,hi=poly.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(cross(a,poly[mid],p)>=0)lo=mid;else hi=mid;}
  return cross(poly[lo],poly[hi],p)>=-1e-10;
}
export function basis(part) {
  const q=quaternion(part.rotation);
  return {p:vec(part.position),r:new Vector3(1,0,0).applyQuaternion(q),u:new Vector3(0,1,0).applyQuaternion(q),f:new Vector3(0,0,1).applyQuaternion(q)};
}
export function centerLocal(optics) { return new Vector3(0,optics.offset/100*9/16/(2*optics.throwRatio),1).normalize(); }
export function aimProjector(state) {
  const d=vec((state.preMirror||state.foldMirror||state.mirror).position).sub(vec(state.projector.position)).normalize();
  // Roll is around the image's center ray; keep the image centered on the mirror.
  const q=new Quaternion().setFromUnitVectors(centerLocal(state.optics),d);
  q.premultiply(new Quaternion().setFromAxisAngle(d, state.aimRoll*Math.PI/180));
  state.projector.rotation=degrees(q);
}
export function defaultState() {
  const s={version:1,projector:{position:[16,0,20],rotation:[0,0,0]},mirror:{position:[-12.47,0,155],rotation:[0,135,0],width:110,height:115},optics:{throwRatio:1.2,offset:100,lumens:220,reflectance:.9,transmission:.65},aimRoll:0,source:{kind:'live',rotation:0,flipX:false,flipY:false},display:{exposure:1.4,rays:true,view:'outside'},modelScale:1};
  // Mirror's local +Z points toward the projector / face. +X is lateral.
  s.mirror.rotation=degrees(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(new Vector3(0,1,0),new Vector3(Math.SQRT1_2,0,Math.SQRT1_2),new Vector3(Math.SQRT1_2,0,-Math.SQRT1_2))));
  aimProjector(s); return s;
}
export function rayForUV(state,u,v) {
  const p=basis(state.projector), w=1/state.optics.throwRatio,h=w*9/16;
  return new Ray(p.p,p.f.clone().addScaledVector(p.r,(u-.5)*w).addScaledVector(p.u,(v-.5+state.optics.offset/200)*h).normalize());
}
export function mirrorHit(ray,mirror) {
  const m=basis(mirror);
  if(mirror.curvatureRadius){const R=mirror.curvatureRadius,center=m.p.clone().addScaledVector(m.f,-R),oc=ray.origin.clone().sub(center),b=oc.dot(ray.direction),disc=b*b-oc.lengthSq()+R*R;if(disc<0)return null;for(const t of [-b-Math.sqrt(disc),-b+Math.sqrt(disc)]){if(t<=.001)continue;const point=ray.at(t,new Vector3()),normal=point.clone().sub(center).divideScalar(R),rel=point.clone().sub(m.p);if(normal.dot(m.f)<=0||ray.direction.dot(normal)>=0||!mirrorContains(mirror,rel.dot(m.r),rel.dot(m.u)))continue;return{point,distance:t,direction:reflect(ray.direction,normal)};}return null;}
  const den=ray.direction.dot(m.f);
  if (den>=-1e-8) return null; // Single front-surface mirror.
  const t=m.p.clone().sub(ray.origin).dot(m.f)/den;
  if(t<=.001) return null;
  const point=ray.at(t,new Vector3()),q=point.clone().sub(m.p);
  if(!mirrorContains(mirror,q.dot(m.r),q.dot(m.u))) return null;
  return {point,distance:t,direction:reflect(ray.direction,m.f)};
}
function singleProjectionPath(state,point) {
  const p=basis(state.projector),m=basis(state.mirror);
  if(state.mirror.curvatureRadius){const solution=curvedReflection(p.p,point,m,state.mirror.curvatureRadius);if(!solution||!mirrorContains(state.mirror,solution.u,solution.v))return null;const d=solution.hit.clone().sub(p.p),depth=d.dot(p.f);if(depth<=0)return null;const u=.5+d.dot(p.r)/depth*state.optics.throwRatio,v=.5-state.optics.offset/200+d.dot(p.u)/depth*state.optics.throwRatio*16/9;if(u<0||u>1||v<0||v>1)return null;return{u,v,hit:solution.hit,distance:d.length()+point.distanceTo(solution.hit),firstLength:d.length(),secondLength:point.distanceTo(solution.hit)};}
  if(p.p.clone().sub(m.p).dot(m.f)<=0 || point.clone().sub(m.p).dot(m.f)<=0) return null;
  const virtual=reflectPoint(p.p,m.p,m.f),toward=point.clone().sub(virtual),distance=toward.length();
  const t=m.p.clone().sub(virtual).dot(m.f)/toward.dot(m.f);
  if(!(t>0 && t<1)) return null;
  const hit=virtual.clone().addScaledVector(toward,t),rel=hit.clone().sub(m.p);
  if(!mirrorContains(state.mirror,rel.dot(m.r),rel.dot(m.u))) return null;
  const d=hit.clone().sub(p.p),depth=d.dot(p.f);
  if(depth<=0) return null;
  const u=.5+d.dot(p.r)/depth*state.optics.throwRatio;
  const v=.5-state.optics.offset/200+d.dot(p.u)/depth*state.optics.throwRatio*16/9;
  if(u<0||u>1||v<0||v>1) return null;
  return {u,v,hit,distance,firstLength:d.length(),secondLength:point.distanceTo(hit),virtual};
}
export function projectionPath(state,point) {
  const key=state.preMirror?'preMirror':state.foldMirror?'foldMirror':null;
  if(!key)return singleProjectionPath(state,point);
  const p=basis(state.projector),f=basis(state[key]),virtual=reflectPoint(p.p,f.p,f.f),rr=reflect(p.r,f.f).negate(),uu=reflect(p.u,f.f),ff=reflect(p.f,f.f);
  const virtualState={...state,projector:{position:virtual.toArray(),rotation:degrees(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(rr,uu,ff)))}};
  delete virtualState[key];
  const path=projectionPath(virtualState,point);if(!path)return null;
  const next=path.legs?path.legs[1]:path.hit;
  const delta=next.clone().sub(virtual),t=f.p.clone().sub(virtual).dot(f.f)/delta.dot(f.f);if(!(t>0&&t<1))return null;
  const hit=virtual.clone().addScaledVector(delta,t),d=hit.clone().sub(p.p),bounce=mirrorHit(new Ray(p.p,d.clone().normalize()),state[key]);
  if(!bounce||bounce.point.distanceTo(hit)>.001||bounce.direction.distanceTo(next.clone().sub(hit).normalize())>.00001)return null;
  const legs=[p.p,hit,...(path.legs?path.legs.slice(1):[path.hit,point])];
  return {...path,u:1-path.u,[key==='preMirror'?'preHit':'foldHit']:hit,legs,distance:legs.slice(1).reduce((n,v,i)=>n+v.distanceTo(legs[i]),0)};
}
export function engineBlocks(state,a,b) {
  const inv=new Matrix4().compose(vec(state.projector.position),quaternion(state.projector.rotation),new Vector3(1,1,1)).invert(),p=a.clone().applyMatrix4(inv),q=b.clone().applyMatrix4(inv),d=q.sub(p),len=d.length();
  if(['ti-engine-rd','ti-kit-rd','tripo-projector'].includes(state.projector.geometry)){const cadEngineBVH=cadEngineBVHs[state.projector.geometry];if(!cadEngineBVH)throw Error('TI engine CAD geometry is not loaded');return blocked(cadEngineBVH,p,p.clone().add(d));}
  const box=new Box3(new Vector3(-28.485,-17.5,-82.175),new Vector3(28.485,3.5,-1.825));
  const hit=new Ray(p,d.divideScalar(len)).intersectBox(box,new Vector3());return !!hit&&p.distanceTo(hit)<len-.08;
}
export function mirrorBlocks(mirror,a,b) {
  const m=basis(mirror),d=b.clone().sub(a),len=d.length(),ray=new Ray(a,d.divideScalar(len));
  if(mirror.curvatureRadius){const R=mirror.curvatureRadius,c=m.p.clone().addScaledVector(m.f,-R),oc=a.clone().sub(c),bb=oc.dot(ray.direction),disc=bb*bb-oc.lengthSq()+R*R;if(disc<0)return false;for(const t of[-bb-Math.sqrt(disc),-bb+Math.sqrt(disc)]){if(t<=.1||t>=len-.1)continue;const h=ray.at(t,new Vector3()).sub(m.p);if(h.dot(m.f)>-R&&mirrorContains(mirror,h.dot(m.r),h.dot(m.u)))return true;}return false;}
  const t=m.p.clone().sub(a).dot(m.f)/ray.direction.dot(m.f);if(!(t>.1&&t<len-.1))return false;const h=ray.at(t,new Vector3()).sub(m.p);return mirrorContains(mirror,h.dot(m.r),h.dot(m.u));
}
export function blocked(bvh,origin,target,epsilon=.08) {
  const d=target.clone().sub(origin),length=d.length();
  if(length<=epsilon*2) return false;
  return !!bvh.raycastFirst(new Ray(origin,d.divideScalar(length)),DoubleSide,epsilon,length-epsilon);
}
export function visiblePath(state,point,headBVH,faceBVH,path=projectionPath(state,point)) {
  if(!path) return false;
  const p=vec(state.projector.position),legs=path.legs||[p,path.hit,point];
  for(let i=1;i<legs.length;i++){
    const a=legs[i-1],b=legs[i];if(apertureBladeBlocks(state,a,b)||blocked(headBVH,a,b)||i<legs.length-1&&blocked(faceBVH,a,b)||engineBlocks(state,a,b))return false;
    if([state.preMirror,state.foldMirror,state.mirror].filter(Boolean).some(m=>mirrorBlocks(m,a,b)))return false;
  }
  const delta=point.clone().sub(path.hit),len=delta.length(),ray=new Ray(path.hit,delta.divideScalar(len));
  // The final continuous layer of face material transmits diffuse light.
  // Any earlier entry+exit pair is a separate obstructing part of the face.
  const hits=faceBVH.raycast(ray,DoubleSide,.08,len-.10).sort((a,b)=>a.distance-b.distance);
  const unique=hits.filter((h,i)=>i===0||Math.abs(h.distance-hits[i-1].distance)>.08);
  return unique.length===0 || (unique.length===1 && unique[0].face.normal.dot(ray.direction)<0);
}
export function traceRay(state,u,v,headBVH,faceBVH) {
  const initial=rayForUV(state,u,v),points=[initial.origin];let ray=initial,total=0;
  for(const mirror of [state.preMirror,state.foldMirror,state.mirror].filter(Boolean)){
    const hit=mirrorHit(ray,mirror);if(!hit)return {kind:'mirror-miss',points:[...points,ray.at(170,new Vector3())]};
    const shellHits=[headBVH,faceBVH].map(b=>b.raycastFirst(ray,DoubleSide,.08,hit.distance-.08)).filter(Boolean).sort((a,b)=>a.distance-b.distance);
    if(shellHits.length)return {kind:'blocked',points:[...points,shellHits[0].point]};
    if(apertureBladeBlocks(state,ray.origin,hit.point)||engineBlocks(state,ray.origin,hit.point)||[state.preMirror,state.foldMirror,state.mirror].filter(Boolean).some(m=>mirrorBlocks(m,ray.origin,hit.point)))return {kind:'blocked',points:[...points,hit.point]};
    points.push(hit.point);total+=hit.distance;ray=new Ray(hit.point,hit.direction);
  }
  const head=headBVH.raycastFirst(ray,DoubleSide,.08,1500),face=faceBVH.raycastFirst(ray,DoubleSide,.08,1500),hit=face&&(!head||face.distance<head.distance)?face:head;
  if(!hit)return {kind:'escape',points:[...points,ray.at(200,new Vector3())]};
  const obstructed=apertureBladeBlocks(state,ray.origin,hit.point)||engineBlocks(state,ray.origin,hit.point)||[state.preMirror,state.foldMirror,state.mirror].filter(Boolean).some(m=>mirrorBlocks(m,ray.origin,hit.point));
  return {kind:!obstructed&&hit===face?'face':'blocked',points:[...points,hit.point],distance:total+hit.distance};
}

export function moveAlongLine(origin,axis,distance) {return vec(origin).addScaledVector(vec(axis).normalize(),distance).toArray();}
export function validateState(raw) {
  if(!raw||raw.version!==1) throw Error('対応する設定ファイルではありません');
  const s=structuredClone(raw), finite=(v,min,max)=>Number.isFinite(v)&&v>=min&&v<=max;
  for(const part of [s.projector,s.mirror,s.foldMirror,s.preMirror].filter(Boolean)) {
    if(!part||!Array.isArray(part.position)||part.position.length!==3||!part.position.every(v=>finite(v,-3000,3000))||!Array.isArray(part.rotation)||part.rotation.length!==3||!part.rotation.every(v=>finite(v,-360,360))) throw Error('位置または角度が範囲外です');
  }
  if(s.apertureBlade){const b=s.apertureBlade;if(!finite(b.size,2,50)||!finite(b.cutDistance,.1,b.size/Math.SQRT2-.01)||!finite(b.thickness,.1,3)||!finite(b.distance,b.thickness/2+.1,30))throw Error('遮光片の寸法が不正です');}
  if(s.projector.geometry!==undefined&&!['ti-engine-rd','ti-kit-rd','tripo-projector'].includes(s.projector.geometry))throw Error('Unknown engine geometry');
  if(s.preMirror&&(!s.foldMirror||s.preMirror.curvatureRadius||!finite(s.preMirror.width,1,1000)||!finite(s.preMirror.height,1,1000)))throw Error("Invalid first folding mirror");
  if(s.foldMirror&&(!finite(s.foldMirror.width,1,1000)||!finite(s.foldMirror.height,1,1000)||s.foldMirror.curvatureRadius))throw Error('折り返し鏡は平面鏡にしてください');
  if(!finite(s.mirror.width,1,1000)||!finite(s.mirror.height,1,1000)) throw Error('ミラー寸法が範囲外です');
  if(s.mirror.curvatureRadius!==undefined&&(!finite(s.mirror.curvatureRadius,10,5000)||!s.mirror.outline&&Math.hypot(s.mirror.width,s.mirror.height)/2>=s.mirror.curvatureRadius))throw Error('曲面鏡の曲率半径または開口寸法が不正です');
  if(s.design!==undefined&&!['original','full-coverage-v1','internal-fold-v1','cad-internal-v1','adult-kit-v1','compact-optical-v2'].includes(s.design))throw Error('未対応のシェル設計です');
  for(const mirror of [s.mirror,s.foldMirror,s.preMirror].filter(Boolean))if(mirror.outline!==undefined){
    const p=mirror.outline;
    if(!Array.isArray(p)||p.length<3||p.length>4096||!p.every(v=>Array.isArray(v)&&v.length===2&&v.every(x=>finite(x,-.500001,.500001))))throw Error('ミラー輪郭が不正です');
    let area=0;for(let i=0;i<p.length;i++){const a=p[i],b=p[(i+1)%p.length],c=p[(i+2)%p.length];area+=a[0]*b[1]-a[1]*b[0];if((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])<-1e-10)throw Error('ミラー輪郭は反時計回りの凸多角形にしてください');}if(area<=0)throw Error('ミラー輪郭の面積がありません');if(mirror.curvatureRadius&&p.some(([x,y])=>Math.hypot(x*mirror.width,y*mirror.height)>=mirror.curvatureRadius))throw Error('Mirror outline exceeds curvature radius');
  }
  if(!s.optics||!finite(s.optics.throwRatio,.2,5)||!finite(s.optics.offset,-200,200)||!finite(s.optics.lumens,1,1000)||!finite(s.optics.reflectance,0,1)||!finite(s.optics.transmission,0,1)) throw Error('光学設定が範囲外です');
  if(!finite(s.modelScale,.01,100)||!finite(s.aimRoll,-360,360)) throw Error('スケールまたはロールが範囲外です');
  if(!s.source||!['live','pattern','video'].includes(s.source.kind)||![0,90,180,270].includes(s.source.rotation)||typeof s.source.flipX!=='boolean'||typeof s.source.flipY!=='boolean') throw Error('映像設定が不正です');
  if(!s.display||!finite(s.display.exposure,.1,5)||!['outside','inside'].includes(s.display.view)||typeof s.display.rays!=='boolean') throw Error('表示設定が不正です');
  return s;
}
