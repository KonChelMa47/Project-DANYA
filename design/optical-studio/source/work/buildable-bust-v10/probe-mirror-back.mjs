import fs from 'node:fs';import * as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';import {obs} from './route-check.mjs';
const d=JSON.parse(fs.readFileSync('work/buildable-bust-v8/datums.json')).mirror.matrix,m=new T.Matrix4().fromArray(d),head=obs.find(o=>o.name==='head'),out=[];
for(let x of [-40,-25,-10,0,10,25,40])for(let y of [-20,0,20]){
 const a=new T.Vector3(x,y,-16).applyMatrix4(m),dir=new T.Vector3(0,0,-1).transformDirection(m),hit=head.g.boundsTree.raycastFirst(new T.Ray(a,dir),T.DoubleSide);
 out.push({local:[x,y,-16],start:a.toArray(),hit:hit?.point.toArray(),distance:hit?.distance});
}
console.log(JSON.stringify(out,null,2));fs.writeFileSync('work/buildable-bust-v10/mirror-head-distances.json',JSON.stringify(out,null,2));
