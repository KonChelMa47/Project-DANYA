import {Shape,ExtrudeGeometry,Matrix4,Quaternion,Euler,Vector3,Ray,DoubleSide}from 'three';
import {MeshBVH}from 'three-mesh-bvh';
const cache=new WeakMap();
export function apertureBladeGeometry(spec){
 const half=spec.size/2,end=half-spec.cutDistance*Math.SQRT2;
 return new ExtrudeGeometry(new Shape([{x:-half,y:-half},{x:end,y:-half},{x:-half,y:end}]),{depth:spec.thickness,bevelEnabled:false,steps:1}).translate(0,0,spec.distance-spec.thickness/2);
}
export function apertureBladeBVH(state){
 if(!state.apertureBlade)return null;
 const signature=JSON.stringify([state.projector.position,state.projector.rotation,state.apertureBlade]);let entry=cache.get(state);
 if(!entry||entry.signature!==signature){const q=new Quaternion().setFromEuler(new Euler(...state.projector.rotation.map(x=>x*Math.PI/180),'XYZ')),g=apertureBladeGeometry(state.apertureBlade).applyMatrix4(new Matrix4().compose(new Vector3(...state.projector.position),q,new Vector3(1,1,1)));entry={signature,bvh:new MeshBVH(g)};cache.set(state,entry);}
 return entry.bvh;
}
export function apertureBladeBlocks(state,a,b){
 const bvh=apertureBladeBVH(state);if(!bvh)return false;const d=b.clone().sub(a),length=d.length();if(length<=.16)return false;
 return !!bvh.raycastFirst(new Ray(a,d.divideScalar(length)),DoubleSide,.08,length-.08);
}
