import fs from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {Vector3,Quaternion,Matrix4,Ray,BoxGeometry,BufferGeometry,Float32BufferAttribute} from 'three';
import {MeshBVH} from 'three-mesh-bvh';
import {reflect,reflectPoint,defaultState,rayForUV,mirrorHit,projectionPath,visiblePath,blocked,moveAlongLine,validateState,basis,mirrorContains,degrees,engineBlocks,setCADEngineGeometry} from '../src/optics.js';
const near=(a,b,e=1e-6)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
const V=(x,y,z)=>new Vector3(x,y,z);
function box(x,y,z,sx,sy,sz){const g=new BoxGeometry(sx,sy,sz);g.translate(x,y,z);return new MeshBVH(g);}
test('single reflection obeys equal angles and is involutory',()=>{
 const n=V(Math.SQRT1_2,0,-Math.SQRT1_2),incoming=V(0,0,1),out=reflect(incoming,n);
 near(out.x,1);near(out.z,0);near(incoming.dot(n),-out.dot(n));near(reflect(out,n).distanceTo(incoming),0);
});
test('100% offset puts lower image edge on optical axis, TR sets horizontal extent',()=>{
 const s=defaultState();s.projector={position:[0,0,0],rotation:[0,0,0]};
 const bottom=rayForUV(s,.5,0);near(bottom.direction.y,0);
 const right=rayForUV(s,1,.5),top=rayForUV(s,.5,1);near(right.direction.x/right.direction.z,1/2.4);near(top.direction.y/top.direction.z,9/16/1.2);
});
test('default center ray reaches the finite mirror center',()=>{
 const s=defaultState(),hit=mirrorHit(rayForUV(s,.5,.5),s.mirror);assert.ok(hit);near(hit.point.distanceTo(new Vector3(...s.mirror.position)),0);
});
test('forward bounce and inverse projection agree at arbitrary pixels',()=>{
 const s=defaultState();s.mirror.width=s.mirror.height=1000;
 for(const [u,v]of [[.2,.3],[.8,.7],[.5,.5]]){
  const hit=mirrorHit(rayForUV(s,u,v),s.mirror);assert.ok(hit);const q=hit.point.clone().addScaledVector(hit.direction,75),path=projectionPath(s,q);assert.ok(path);near(path.u,u);near(path.v,v);near(path.distance,hit.distance+75);
 }
});
test('mirror is finite and one-sided',()=>{
 const s=defaultState();s.mirror.width=s.mirror.height=2;
 assert.equal(mirrorHit(rayForUV(s,.01,.01),s.mirror),null);
 const m=basis(s.mirror);assert.equal(mirrorHit(new Ray(m.p.clone().addScaledVector(m.f,-20),m.f),s.mirror),null);
});
test('head blocks either leg even when displayed as a cutaway',()=>{
 const s=defaultState(),m=mirrorHit(rayForUV(s,.5,.5),s.mirror),q=m.point.clone().addScaledVector(m.direction,90),far=box(2000,2000,2000,1,1,1),p=new Vector3(...s.projector.position);
 const h1=p.clone().lerp(m.point,.5),h2=m.point.clone().lerp(q,.5);
 assert.equal(visiblePath(s,q,far,far),true);
 assert.equal(visiblePath(s,q,box(...h1.toArray(),5,5,5),far),false);
 s.display.view='inside';assert.equal(visiblePath(s,q,box(...h2.toArray(),5,5,5),far),false);
});
test('final face layer transmits, an earlier separate layer obstructs',()=>{
 const s=defaultState(),m=mirrorHit(rayForUV(s,.5,.5),s.mirror),q=m.point.clone().addScaledVector(m.direction,90),far=box(2000,2000,2000,1,1,1);
 const g=new BoxGeometry(30,30,1).toNonIndexed();
 g.applyQuaternion(new Quaternion().setFromUnitVectors(V(0,0,1),m.direction));
 g.translate(...q.clone().addScaledVector(m.direction,-.5).toArray());
 assert.equal(visiblePath(s,q,far,new MeshBVH(g)),true);
 const earlier=g.clone();earlier.translate(...m.direction.clone().multiplyScalar(-20).toArray());
 const positions=new Float32Array(g.attributes.position.array.length*2);positions.set(g.attributes.position.array);positions.set(earlier.attributes.position.array,g.attributes.position.array.length);
 const both=new BufferGeometry();both.setAttribute('position',new Float32BufferAttribute(positions,3));
 assert.equal(visiblePath(s,q,far,new MeshBVH(both)),false);
});
test('line movement keeps orthogonal coordinates fixed and uses normalized axes',()=>{
 assert.deepEqual(moveAlongLine([1,2,3],[0,0,4],12),[1,2,15]);
 const moved=moveAlongLine([5,7,9],[1,1,0],Math.sqrt(2));near(moved[0],6);near(moved[1],8);near(moved[2],9);
});
test('layout validation rejects bad dimensions and nonfinite coordinates',()=>{
 assert.deepEqual(validateState(defaultState()),defaultState());
 const s=defaultState();s.mirror.width=-1;assert.throws(()=>validateState(s));
 s.mirror.width=100;s.projector.position[1]=NaN;assert.throws(()=>validateState(s));
});
test('custom mirror clips its corners instead of treating the bounding box as reflective',()=>{
 const m={width:100,height:100,outline:[[-.5,0],[0,-.5],[.5,0],[0,.5]]};
 assert(mirrorContains(m,0,0));assert(mirrorContains(m,20,20));assert(mirrorContains(m,50,0));
 assert.equal(mirrorContains(m,40,40),false);assert.equal(mirrorContains(m,-40,-40),false);
 const s=defaultState();s.mirror.outline=m.outline;validateState(s);s.mirror.outline=m.outline.slice().reverse();assert.throws(()=>validateState(s));
});

test('convex mirror inverse projection and forward spherical reflection agree',()=>{
  const s=defaultState();s.projector.position=[-55,0,-15];s.projector.rotation=[13.226193079758444,5.317994777415015,-1.2533574058179149];s.mirror={position:[-40,0,110],rotation:[-180,48.076998, -90],width:58,height:48,curvatureRadius:60};
  for(const point of [[70,0,140],[45,35,185],[45,-30,70]]){const p=new Vector3(...point),path=projectionPath(s,p);assert.ok(path);const hit=mirrorHit(rayForUV(s,path.u,path.v),s.mirror);assert.ok(hit);assert.ok(hit.point.distanceTo(path.hit)<1e-5);assert.ok(hit.direction.distanceTo(p.sub(hit.point).normalize())<1e-5);}
});

test('folded flat-to-convex route agrees in both directions and clips first mirror',()=>{
 const s=defaultState();s.projector={position:[-23.767325313057125,0,152.5736089332961],rotation:[-179.9804336048427,-29.248183246184446,-89.99291200106471]};
 s.foldMirror={position:[-45.892857142857146,0,70],rotation:[0,7.517,0],width:200,height:200};
 // Construct known planar fold from its two conjugate lens positions.
 const actual=new Vector3(...s.projector.position),virtual=V(-55,0,-15),normal=actual.clone().sub(virtual).normalize(),right=new Vector3().crossVectors(V(0,1,0),normal).normalize(),up=new Vector3().crossVectors(normal,right);
 s.foldMirror.rotation=degrees(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(right,up,normal)));
 const M=V(-40,0,125),n=V(55,0,135).sub(M).normalize().sub(M.clone().sub(virtual).normalize()).normalize(),mr=V(0,1,0),mu=new Vector3().crossVectors(n,mr);
 s.mirror={position:M.toArray(),rotation:degrees(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(mr,mu,n))),width:50,height:40,curvatureRadius:45};
 const target=V(65,0,135),path=projectionPath(s,target);assert.ok(path);const first=mirrorHit(rayForUV(s,path.u,path.v),s.foldMirror);assert.ok(first);near(first.point.distanceTo(path.foldHit),0);const second=mirrorHit(new Ray(first.point,first.direction),s.mirror);assert.ok(second);near(second.point.distanceTo(path.hit),0);near(second.direction.distanceTo(target.clone().sub(second.point).normalize()),0);near(path.distance,first.distance+second.distance+target.distanceTo(second.point));
 s.foldMirror.width=.01;assert.equal(projectionPath(s,target),null);
});
test('engine envelope casts a shadow and does not obstruct rays in front of lens',()=>{const s=defaultState();s.projector={position:[0,0,0],rotation:[0,0,0]};assert.equal(engineBlocks(s,V(-100,-7,-42),V(100,-7,-42)),true);assert.equal(engineBlocks(s,V(-100,-7,10),V(100,-7,10)),false);});

test('CAD engine geometry replaces the approximate box in occlusion tests',()=>{const s=defaultState();s.projector={position:[0,0,0],rotation:[0,0,0],geometry:'ti-engine-rd'};setCADEngineGeometry(new BoxGeometry(2,2,2).translate(50,0,10));assert.equal(engineBlocks(s,V(40,0,10),V(60,0,10)),true);delete s.projector.geometry;assert.equal(engineBlocks(s,V(40,0,10),V(60,0,10)),false);});

test('three finite reflections preserve inverse and forward ray paths',()=>{
 const s=JSON.parse(fs.readFileSync(new URL('./three-mirror-fixture.json',import.meta.url)));
 for(const q of [V(80,0,135),V(65,25,170),V(65,-25,110)]){
  const p=projectionPath(s,q);assert.ok(p);assert.equal(p.legs.length,5);
  let ray=rayForUV(s,p.u,p.v);
  for(const [i,m] of [s.preMirror,s.foldMirror,s.mirror].entries()){
   const h=mirrorHit(ray,m);assert.ok(h);near(h.point.distanceTo(p.legs[i+1]),0,1e-5);ray=new Ray(h.point,h.direction);
  }
  near(ray.distanceToPoint(q),0,1e-5);
 }
 s.preMirror.width=.01;assert.equal(projectionPath(s,V(80,0,135)),null);
});

test('each folding mirror accepts a convex aperture and rejects invalid winding or concavity',()=>{
 const s=JSON.parse(fs.readFileSync(new URL('./three-mirror-fixture.json',import.meta.url)));
 const polygon=[[-.5,-.5],[.5,-.5],[.3,.5],[-.3,.5]];
 for(const key of ['preMirror','foldMirror']){
  const valid=structuredClone(s);valid[key].outline=polygon;
  assert.deepEqual(validateState(valid)[key].outline,polygon);
  assert.equal(mirrorContains(valid[key],valid[key].width*.49,valid[key].height*.49),false);
  const reversed=structuredClone(valid);reversed[key].outline.reverse();assert.throws(()=>validateState(reversed));
  const concave=structuredClone(valid);concave[key].outline=[[-.5,-.5],[.5,-.5],[0,0],[.5,.5],[-.5,.5]];assert.throws(()=>validateState(concave));
 }
});
