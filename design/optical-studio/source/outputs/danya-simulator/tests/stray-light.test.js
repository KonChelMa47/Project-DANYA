import test from 'node:test';import assert from 'node:assert/strict';
import{BoxGeometry}from 'three';import{MeshBVH}from 'three-mesh-bvh';
import{createStrayTracer}from '../src/stray-light.js';import{maskAt,maskSignature}from '../src/projection-mask.js';
const bvh=(x,y,z,w=20)=>new MeshBVH(new BoxGeometry(w,w,1).translate(x,y,z));
const state={projector:{position:[0,0,0],rotation:[0,0,0]},mirror:{position:[100,0,20],rotation:[0,180,0],width:10,height:10},optics:{throwRatio:1.2,offset:0},modelScale:1};
test('non-sequential trace detects direct face light when the planned mirror is missed',()=>{const trace=createStrayTracer(state,bvh(100,0,50),bvh(0,0,50),new BoxGeometry(1,1,1).translate(100,100,100));const r=trace(.5,.5);assert.equal(r.kind,'face');assert.deepEqual(r.route,[]);});
test('nearest opaque head blocks a direct ghost route',()=>{const trace=createStrayTracer(state,bvh(0,0,20),bvh(0,0,50),new BoxGeometry(1,1,1).translate(100,100,100));assert.equal(trace(.5,.5).kind,'head');});
test('projector mask indexes physical UV rows, rejects disabled and out-of-range pixels',()=>{const m=new Uint8Array(1280*720);m[20*1280+10]=255;assert.ok(maskAt(m,10.5/1280,20.5/720));assert.ok(!maskAt(m,11.5/1280,20.5/720));assert.ok(!maskAt(m,1,1));});
test('mask signature changes with pose but not video selection',()=>{assert.equal(maskSignature(state),maskSignature({...state,source:{kind:'video'}}));assert.notEqual(maskSignature(state),maskSignature({...state,projector:{...state.projector,position:[1,0,0]}}));});
