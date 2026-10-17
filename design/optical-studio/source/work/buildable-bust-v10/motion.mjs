import *as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import{pivot,vertices,gap,profile}from'../buildable-bust-v4/motion.mjs';export{pivot,vertices,gap,profile};
export const pitchPivot=new T.Vector3(pivot.x,pivot.y,-4);
export function rotation(y,p,r){return {y,p,r};}
function at(c,axis,a){return new T.Matrix4().makeTranslation(...c.toArray()).multiply(new T.Matrix4().makeRotationAxis(axis,a*Math.PI/180)).multiply(new T.Matrix4().makeTranslation(...c.clone().negate().toArray()));}
export function around({y,p,r}){return at(pivot,new T.Vector3(0,0,1),y).multiply(at(pitchPivot,new T.Vector3(0,1,0),p)).multiply(at(pivot,new T.Vector3(1,0,0),r));}
