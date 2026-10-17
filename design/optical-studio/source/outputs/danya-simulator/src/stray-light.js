import {apertureBladeBVH}from './aperture-blade.js';
import {Vector3,Ray,DoubleSide,Matrix4} from 'three';
import {MeshBVH} from 'three-mesh-bvh';
import {rayForUV,mirrorHit,vec,quaternion} from './optics.js';
import {mirrorSolidGeometry} from './mirror-geometry.js';
// Non-sequential trace: the nearest physical surface wins, independently of the intended route.
// Opaque surfaces absorb; front-surface mirrors specularly reflect. Face scattering/secondary reflections are not modeled.
export function createStrayTracer(state,headBVH,faceBVH,projectorGeometry) {
 const matrix=p=>new Matrix4().compose(vec(p.position),quaternion(p.rotation),new Vector3(1,1,1));
 const body=projectorGeometry.clone().applyMatrix4(matrix(state.projector));const bodyBVH=new MeshBVH(body);
 const blade=apertureBladeBVH(state);
 const mirrors=['preMirror','foldMirror','mirror'].filter(k=>state[k]).map(key=>{const part=state[key],g=mirrorSolidGeometry(part).scale(part.width,part.height,1).translate(0,0,-.85).applyMatrix4(matrix(part));return{key,part,bvh:new MeshBVH(g)};});
 return (u,v,maxBounces=8,initialRay=null)=>{
  let ray=initialRay||rayForUV(state,u,v),route=[],points=[ray.origin.clone()];
  for(let bounce=0;bounce<=maxBounces;bounce++){
   let nearest=null;const consider=(kind,hit,extra={})=>{if(hit&&(!nearest||hit.distance<nearest.distance))nearest={kind,...hit,...extra};};
   if(blade)consider('aperture-blade',blade.raycastFirst(ray,DoubleSide,.08,1500));
   consider('head',headBVH.raycastFirst(ray,DoubleSide,.08,1500));consider('face',faceBVH.raycastFirst(ray,DoubleSide,.08,1500));consider('body',bodyBVH.raycastFirst(ray,DoubleSide,.08,1500));
   for(const m of mirrors){consider('mirror-back',m.bvh.raycastFirst(ray,DoubleSide,.08,1500),{key:m.key});consider('mirror',mirrorHit(ray,m.part),{key:m.key});}
   if(!nearest)return{kind:'escape',route,points};points.push(nearest.point.clone());
   if(nearest.kind==='face'){
    const further=faceBVH.raycast(ray,DoubleSide,nearest.distance+.08,1500).sort((a,b)=>a.distance-b.distance);
    // Only use the immediately following outward crossing as the exterior destination.
    const outer=nearest.face.normal.dot(ray.direction)>0?nearest: further.find(h=>h.face.normal.dot(ray.direction)>0);
    return{kind:'face',route,points,point:nearest.point,outer:outer?.point||null};
   }
   if(nearest.kind!=='mirror')return{kind:nearest.kind,route,points};
   route.push(nearest.key);if(bounce===maxBounces)return{kind:'bounce-limit',route,points};ray=new Ray(nearest.point,nearest.direction);
  }
 };
}
