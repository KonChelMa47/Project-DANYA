import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{checkRod}from'./route-check.mjs';
const d=JSON.parse(fs.readFileSync('outputs/danya-simulator/assets/buildable-bust-v9/drive-layout.json')).motors[0],m=new T.Matrix4().compose(new T.Vector3(...d.tip),new T.Quaternion(...d.quaternion),new T.Vector3(1,1,1)),start=new T.Vector3(20.5,-48,0).applyMatrix4(m).toArray();const rows=[];
console.log('ROOT',start);
for(let x of[-60,-40,-20,0,20])for(let y of[-85,-60,-30,0,30,60,85]){
 const end=[x,y,-148],r=checkRod(start,end,0,[],5.6);if(!r.hits.length&&!r.lightHits){rows.push({start,end,...r});console.log('PASS',JSON.stringify(rows.at(-1)));}
}
fs.writeFileSync('work/buildable-bust-v10/yaw-frame-probes.json',JSON.stringify({start,rows},null,2));
