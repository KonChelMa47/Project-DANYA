import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{obs,checkRod}from'./route-check.mjs';
const body=obs.find(o=>o.name==='body'),cross=[];for(let y=-60;y<=60;y+=15){const h=body.g.boundsTree.raycastFirst(new T.Ray(new T.Vector3(0,y,-43),new T.Vector3(-1,0,0)),T.DoubleSide);cross.push([h.point.x+7,y,-43]);}
const rows=[];for(let i=1;i<cross.length;i++)rows.push({name:'curved-cross-'+i,start:cross[i-1],end:cross[i],...checkRod(cross[i-1],cross[i],0,[],5.6)});
for(const [name,a,b]of[['rear-arm',[-82,.345745,-35],cross[4]],['front-negative',[4.984,-46.4196,-31],[5,-80,-40]],['negative-to-back',[5,-80,-40],cross[0]]])rows.push({name,start:a,end:b,...checkRod(a,b,0,['yaw-bearing-base'],5.6)});
console.log(JSON.stringify(rows,null,2));fs.writeFileSync('work/buildable-bust-v10/back-negative-probes.json',JSON.stringify(rows,null,2));
