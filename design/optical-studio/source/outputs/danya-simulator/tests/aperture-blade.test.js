import test from 'node:test';import assert from 'node:assert/strict';import{Vector3}from 'three';import{apertureBladeBlocks}from '../src/aperture-blade.js';import{defaultState,validateState}from '../src/optics.js';import{maskSignature}from '../src/projection-mask.js';
test('finite diagonal blade clips a pupil corner, leaves the center open, and follows the projector',()=>{
 const state=defaultState();state.projector.position=[0,0,0];state.projector.rotation=[0,0,0];state.apertureBlade={distance:1,cutDistance:2.4,size:12,thickness:.3};validateState(state);
 assert.equal(apertureBladeBlocks(state,new Vector3(-3,-3,0),new Vector3(-3,-3,4)),true);
 assert.equal(apertureBladeBlocks(state,new Vector3(0,0,0),new Vector3(0,0,4)),false);
 assert.equal(apertureBladeBlocks(state,new Vector3(-7,-7,0),new Vector3(-7,-7,4)),false);
 const before=maskSignature(state);state.projector.position=[100,0,0];assert.equal(apertureBladeBlocks(state,new Vector3(-3,-3,0),new Vector3(-3,-3,4)),false);assert.equal(apertureBladeBlocks(state,new Vector3(97,-3,0),new Vector3(97,-3,4)),true);assert.notEqual(maskSignature(state),before);
 state.apertureBlade.distance=0;assert.throws(()=>validateState(state));
});
