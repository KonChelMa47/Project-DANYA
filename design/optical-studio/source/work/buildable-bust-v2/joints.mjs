import fs from 'node:fs';import{point}from './surface.mjs';
let parts=[],bolts=[];
for(let band=0;band<2;band++)for(let q=0;q<4;q++){
const joints=[],z=band?-58:-143;
for(let edge=0;edge<2;edge++){let t=(q+edge)*Math.PI/2,p=point(z,t,false),rad=[Math.cos(t),Math.sin(t),0],tan=[-Math.sin(t),Math.cos(t),0],sign=edge?-1:1;
let center=[p[0]-7*rad[0]+sign*3.2*tan[0],p[1]-7*rad[1]+sign*3.2*tan[1],z],holeCenter=[p[0]-10*rad[0],p[1]-10*rad[1],z];
joints.push({type:'vertical',center,size:[16,6,18],angle:t,holeCenter,axis:tan});}
for(let frac of[.3,.7]){let t=(q+frac)*Math.PI/2,p=point(-95,t,false),rad=[Math.cos(t),Math.sin(t),0],center=[p[0]-7*rad[0],p[1]-7*rad[1],-95+(band?3.2:-3.2)],holeCenter=[p[0]-10*rad[0],p[1]-10*rad[1],-95];joints.push({type:'horizontal',center,size:[16,18,6],angle:t,holeCenter,axis:[0,0,1]});}
parts.push({name:`shell-${band?'upper':'lower'}-${q}`,joints});}
fs.writeFileSync('work/buildable-bust-v2/joints.json',JSON.stringify({scope:'Static external shell screw-lug prototype. M3 clearance 3.4mm; plain nut/washer inside; intended M3x20 joints. Lug pairs each approx12.4mm grip; external shell unloaded. Interior optical/mechanical interference NOT approved.',parts},null,2));
