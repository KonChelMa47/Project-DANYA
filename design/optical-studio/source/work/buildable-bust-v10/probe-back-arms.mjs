import fs from'node:fs';import{checkRod}from'./route-check.mjs';
const sp=JSON.parse(fs.readFileSync('work/buildable-bust-v10/back-chassis-probes.json')).filter(r=>r.yy===60),rows=[];
for(const r of sp){const start=[-49,r.sign<0?-46.41962:47.11111,-31],end=r.points.at(-2);rows.push({name:'arm-'+r.sign,start,end,...checkRod(start,end,0,['yaw-bearing-base'],5.6)});const foot=[-67,r.sign*60,-150];rows.push({name:'foot-'+r.sign,start:r.points[0],end:foot,...checkRod(r.points[0],foot,0,[],5.6)});}
const a=sp[0].points.at(-1),b=sp[1].points.at(-1);rows.push({name:'crossbar',start:a,end:b,...checkRod(a,b,0,[],5.6)});
const mid=[(a[0]+b[0])/2,0,-28],start=[-76.01616859,.345745,-31];rows.push({name:'middle-arm',start,end:mid,...checkRod(start,mid,0,['yaw-bearing-base'],5.6)});
console.log(JSON.stringify(rows,null,2));fs.writeFileSync('work/buildable-bust-v10/back-arms-probes.json',JSON.stringify(rows,null,2));
