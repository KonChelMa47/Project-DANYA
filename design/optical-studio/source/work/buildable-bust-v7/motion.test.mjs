import assert from'node:assert/strict';import *as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{pivot,pitchPivot,around,rotation}from'./motion.mjs';
function eq(a,b){assert.ok(a.distanceTo(b)<1e-8,JSON.stringify([a.toArray(),b.toArray()]));}
eq(pitchPivot.clone().applyMatrix4(around(rotation(0,90,0))),pitchPivot);
eq(pitchPivot.clone().add(new T.Vector3(100,0,0)).applyMatrix4(around(rotation(0,90,0))),new T.Vector3(pivot.x,pivot.y,-104));
eq(pivot.clone().applyMatrix4(around(rotation(0,90,0))),new T.Vector3(pivot.x+24,pivot.y,-4));
eq(pivot.clone().applyMatrix4(around(rotation(0,0,90))),pivot);
console.log('Separated pitch/roll pivots: known quarter-turn positions passed.');
