import test from 'node:test';import assert from 'node:assert/strict';
import {BufferGeometry,Float32BufferAttribute}from 'three';
import {buildFaceWarpGeometry}from '../src/face-warp.js';
test('the whole frontal image maps into the smaller projector footprint without cropping source UVs',()=>{
 const face=new BufferGeometry().setAttribute('position',new Float32BufferAttribute([0,-70,40,0,70,40,0,70,240,0,-70,240],3)).setIndex([0,1,2,0,2,3]);face.userData.outerIndices=face.index.array;
 const g=buildFaceWarpGeometry(face,new Float32Array([.35,.26,.68,.3,.61,.73,.4,.69]),new Float32Array([1,1,1,1]));
 assert.deepEqual([...g.attributes.uv.array],[0,0,1,0,1,1,0,1]);assert.equal(g.index.count,6);
 assert.ok(Math.abs(g.attributes.position.getX(0)-(-.3))<1e-6);
 assert.ok(Math.abs(g.attributes.position.getY(2)-.46)<1e-6);
 const blocked=buildFaceWarpGeometry(face,new Float32Array(8),new Float32Array([1,1,0,1]));assert.equal(blocked.index.count,0);
});
