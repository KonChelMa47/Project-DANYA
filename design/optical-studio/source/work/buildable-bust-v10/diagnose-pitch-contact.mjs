import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{obs}from'./trial-route-check.mjs';import{around}from'./motion.mjs';
let a=obs.find(o=>o.name==='trial-pitch-support'),rows=[];
if(a)for(let b of obs.filter(o=>o.name==='upper-retaining-plate'||o.name.startsWith('retainer-spacer-'))){for(let y=-30;y<=30;y++){let m=around({y,p:0,r:0});if(b.g.boundsTree.intersectsGeometry(a.g,m)){rows.push({name:b.name,yaw:y});break;}}}console.log(JSON.stringify(rows));
