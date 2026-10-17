import {obs,cylinder} from '../buildable-bust-v10/revised-complete-route-check.mjs';
import *as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
const keep=obs.filter(o=>['head','face','projector','mirror-glass','foldMirror-glass'].includes(o.name)||o.name.includes('printed-mirror')||o.name.includes('printed-foldMirror')||o.name==='drop-projector-tray-t5');
for(const o of keep)console.log(o.name,o.g.boundingBox.min.toArray(),o.g.boundingBox.max.toArray());
for(const y of [-40,40,-85,85]){let g=cylinder([-45,y,-150],[-45,y,40],8);console.log('post',y,keep.filter(o=>o.g.boundsTree.intersectsGeometry(g,new T.Matrix4())).map(o=>o.name));}
