import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{obs,check,cylinder}from'./route-check.mjs';
const head=obs.find(o=>o.name==='head'),rows=[],seams=JSON.parse(fs.readFileSync('work/buildable-bust-v10/magnet-datums.json'));
for(const seam of seams)for(const dx of[12,20,28,36]){
 const[x,y,z]=seam.seam,sign=Math.sign(y),hit=head.g.boundsTree.raycastFirst(new T.Ray(new T.Vector3(x-dx,0,z),new T.Vector3(0,sign,0)),T.DoubleSide);if(!hit)continue;
 const c=hit.point.clone().add(new T.Vector3(0,-sign*5.0,0)),g=cylinder(c.clone().add(new T.Vector3(0,-sign*4,0)).toArray(),c.clone().add(new T.Vector3(0,sign*4,0)).toArray(),4.2),r=check(g,3,[],true);rows.push({id:seam.id,dx,center:c.toArray(),wall:hit.point.toArray(),...r});if(!r.hits.length&&!r.lightHits)console.log('PASS',JSON.stringify(rows.at(-1)));
}
fs.writeFileSync('work/buildable-bust-v10/rear-magnet-search.json',JSON.stringify(rows,null,2));
