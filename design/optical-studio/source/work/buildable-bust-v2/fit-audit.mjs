import fs from 'node:fs';
import {Vector3,Quaternion,Euler} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import{point,profiles}from './surface.mjs';
const layout=JSON.parse(fs.readFileSync('work/torso-flat/torso-v3-layout.json'));
const hull=JSON.parse(fs.readFileSync('work/torso-flat/projector-hull.json')).vertices;
const q=new Quaternion().setFromEuler(new Euler(...layout.projector.rotation.map(x=>x*Math.PI/180),'XYZ'));
const verts=hull.map(v=>new Vector3(...v).applyQuaternion(q).add(new Vector3(...layout.projector.position)));
// Each inner horizontal contour is sampled as a polygon. Outside witnesses disprove fit;
// passing hull vertices alone is not a proof for this nonconvex container.
function signedRadial(v){if(v.z<profiles[0][0]||v.z>profiles.at(-1)[0])return -Infinity;let t=Math.atan2(v.y,(v.x-15.7629));let lo=t-.5,hi=t+.5;for(let k=0;k<28;k++){let m=(lo+hi)/2,p=point(v.z,m,true),angle=Math.atan2(p[1],p[0]-15.7629);while(angle-t>Math.PI)angle-=Math.PI*2;while(angle-t< -Math.PI)angle+=Math.PI*2;if(angle<t)lo=m;else hi=m;}const p=point(v.z,(lo+hi)/2,true);return Math.hypot(p[0]-15.7629,p[1])-Math.hypot(v.x-15.7629,v.y);}
let reports=[];for(const range of [[0,0,0],[15,10,8],[30,15,10],[45,20,15]]){let worst=Infinity,witness=null,outside=0,total=0;for(let yaw=-range[0];yaw<=range[0];yaw+=range[0]?5:1)for(let pitch=-range[1];pitch<=range[1];pitch+=range[1]?5:1)for(let roll=-range[2];roll<=range[2];roll+=range[2]?2:1){const rot=new Quaternion().setFromEuler(new Euler(roll*Math.PI/180,pitch*Math.PI/180,yaw*Math.PI/180,'ZYX'));for(const v of verts){const w=v.clone().sub(new Vector3(5,0,0)).applyQuaternion(rot).add(new Vector3(5,0,0)),gap=signedRadial(w);total++;if(gap<0)outside++;if(gap<worst){worst=gap;witness={pose:[yaw,pitch,roll],point:w.toArray()};}}}reports.push({rangeDeg:range,minRadialGapMM:worst,outsideSamples:outside,total,witness});}
const report={scope:'Screen only: negative radial gap is a protrusion witness. No continuous clearance or positive-fit guarantee. Full-size projector unchanged; shells and optics unchanged.',reports};fs.writeFileSync('work/buildable-bust-v2/fit-audit.json',JSON.stringify(report,null,2));console.log(report);
