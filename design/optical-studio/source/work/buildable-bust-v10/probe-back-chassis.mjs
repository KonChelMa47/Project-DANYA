import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{obs,checkRod}from'./route-check.mjs';
const body=obs.find(o=>o.name==='body'),rows=[];
for(const sign of[-1,1])for(const yy of[60,70,80]){const points=[];for(let z=-148;z<=-28;z+=15){const h=body.g.boundsTree.raycastFirst(new T.Ray(new T.Vector3(0,sign*yy,z),new T.Vector3(-1,0,0)),T.DoubleSide);if(h)points.push([h.point.x+7,sign*yy,z]);}
 let checks=[];for(let i=1;i<points.length;i++)checks.push(checkRod(points[i-1],points[i],0,[],5.6));const row={sign,yy,points,checks};rows.push(row);console.log(JSON.stringify(row));}
fs.writeFileSync('work/buildable-bust-v10/back-chassis-probes.json',JSON.stringify(rows,null,2));
