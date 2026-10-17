import fs from'node:fs';import *as T from'../../outputs/danya-simulator/node_modules/three/build/three.module.js';import{MeshBVH}from'../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';import{obs,check,checkRod}from'./route-check.mjs';
const d=JSON.parse(fs.readFileSync('outputs/danya-simulator/assets/buildable-bust-v9/drive-layout.json')).motors[0],q=JSON.parse(fs.readFileSync('outputs/danya-simulator/assets/buildable-bust-v9/backplate-motor-search.json')).rows.find(r=>r.role==='yaw'&&r.deg===120).q;
const old=new T.Matrix4().compose(new T.Vector3(...d.tip),new T.Quaternion(...d.quaternion),new T.Vector3(1,1,1)),now=new T.Matrix4().compose(new T.Vector3(...d.tip),new T.Quaternion(...q),new T.Vector3(1,1,1)),delta=now.clone().multiply(old.invert());
const names=obs.filter(o=>o.name==='motor-yaw'||o.name.startsWith('selected-backmount-yaw')).map(o=>o.name);
for(const o of obs.filter(o=>names.includes(o.name))){o.g.applyMatrix4(delta);o.g.boundsTree=new MeshBVH(o.g);}
const rows=[{name:'motor-body',...check(obs.find(o=>o.name==='motor-yaw').g,0,[...names,'horn-clutch-hub'],true)}];
for(const row of JSON.parse(fs.readFileSync('work/buildable-bust-v10/back-negative-probes.json')).filter(r=>r.name.startsWith('curved-cross')||r.name==='rear-arm'))rows.push({...row,...checkRod(row.start,row.end,0,['yaw-bearing-base'],5.6)});
const root=new T.Vector3(20.5,-50,0).applyMatrix4(now).toArray();for(const end of[[-105,0,-73],[-103,-15,-73],[-100,0,-58],[-105,0,-88]])rows.push({name:'motor-arm',start:root,end,...checkRod(root,end,0,[],5.6)});
console.log(JSON.stringify(rows,null,2));fs.writeFileSync('work/buildable-bust-v10/yaw120-probes.json',JSON.stringify({quaternion:q,rows},null,2));
