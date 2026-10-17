import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{obs,checkRod}from'./route-check.mjs';
const body=obs.find(o=>o.name==='body'),rows=[],end=[];for(const sign of[-1,1]){const h=body.g.boundsTree.raycastFirst(new T.Ray(new T.Vector3(0,sign*60,-10),new T.Vector3(-1,0,0)),T.DoubleSide);end.push([h.point.x+7,sign*60,-10]);}
rows.push({name:'top-crossbar',start:end[0],end:end[1],...checkRod(end[0],end[1],0,[],5.6)});
for(let i=0;i<6;i++){const t=i*Math.PI/3,start=[-22.016168594+54*Math.cos(t),.345745+54*Math.sin(t),-7.7];for(const p of[end[0],end[1],[(end[0][0]+end[1][0])/2,0,-10]])rows.push({name:'upper-'+i,start,end:p,...checkRod(start,p,0,['upper-retaining-plate'],5.6)});}
console.log(JSON.stringify(rows,null,2));fs.writeFileSync('work/buildable-bust-v10/back-upper-probes.json',JSON.stringify(rows,null,2));
