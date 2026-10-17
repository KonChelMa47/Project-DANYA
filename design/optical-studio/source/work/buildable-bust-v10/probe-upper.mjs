import fs from 'node:fs';import {checkRod} from './route-check.mjs';
const rows=[];
for(const sign of [-1,1]){
 const start=[-48,sign<0?-29:30,44];
 const end=[-35.2691206961-sign*.2947691184,.6498670941+sign*9.9956546142,226.4191666926];
 for(const mid of [[-75,sign*55,185],[-80,sign*50,205],[-70,sign*40,235],[-65,sign*40,235],[-75,sign*35,240],[-60,sign*35,242],[-65,sign*20,245]]){
  const a=checkRod(start,mid,3,['roll-carrier-v7'],5.6),b=checkRod(mid,end,3,['printed-mirror-fixed-plate','printed-mirror-frame-nut-0','printed-mirror-frame-nut-1'],5.6);
  rows.push({sign,start,mid,end,a,b});console.log(JSON.stringify(rows.at(-1)));
 }
}
fs.writeFileSync('work/buildable-bust-v10/upper-probes.json',JSON.stringify(rows,null,2));
