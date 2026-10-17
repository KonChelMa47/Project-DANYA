import * as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{OBB}from '../../outputs/danya-simulator/node_modules/three/examples/jsm/math/OBB.js';
// For a rigid matrix, transform the local centre as a point, not just by translation.
export function rigidOBB(localCenter,halfSize,matrix){return new OBB(localCenter.clone().applyMatrix4(matrix),halfSize.clone(),new T.Matrix3().setFromMatrix4(matrix));}
