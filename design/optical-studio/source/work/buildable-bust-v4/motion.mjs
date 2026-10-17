import fs from'node:fs';import * as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
const profiles=JSON.parse(fs.readFileSync('outputs/danya-simulator/assets/buildable-bust-v3/design.json')).profiles;
export const pivot=new T.Vector3(-22.01616859436035,.3457450866699219,20);
export function profile(z){let i=0;while(i<profiles.length-2&&z>profiles[i+1][0])i++;let h=profiles[i+1][0]-profiles[i][0],t=(z-profiles[i][0])/h;return[1,2,3].map(k=>{const sec=j=>(profiles[j+1][k]-profiles[j][k])/(profiles[j+1][0]-profiles[j][0]),slope=j=>{if(j===0)return sec(0);if(j===profiles.length-1)return sec(j-1);let a=sec(j-1),b=sec(j);return a*b<=0?0:2*a*b/(a+b);};return(2*t**3-3*t*t+1)*profiles[i][k]+(t**3-2*t*t+t)*h*slope(i)+(-2*t**3+3*t*t)*profiles[i+1][k]+(t**3-t*t)*h*slope(i+1);});}
export function gap(v){if(v.z< -157||v.z>24)return -1000;const[a,b,c]=profile(v.z),rx=a-3,ry=b-3;return (1-Math.hypot((v.x-c)/rx,(v.y-pivot.y)/ry))*Math.min(rx,ry);}
const s=JSON.parse(fs.readFileSync('work/torso-flat/torso-v3-layout.json')),pq=new T.Quaternion().setFromEuler(new T.Euler(...s.projector.rotation.map(x=>x*Math.PI/180),'XYZ'));
export const vertices=JSON.parse(fs.readFileSync('work/torso-flat/projector-hull.json')).vertices.map(v=>new T.Vector3(...v).applyQuaternion(pq).add(new T.Vector3(...s.projector.position)));
export function rotation(y,p,r){return new T.Quaternion().setFromEuler(new T.Euler(r*Math.PI/180,p*Math.PI/180,y*Math.PI/180,'ZYX'));}
export function around(q){return new T.Matrix4().makeTranslation(...pivot.toArray()).multiply(new T.Matrix4().makeRotationFromQuaternion(q)).multiply(new T.Matrix4().makeTranslation(...pivot.clone().negate().toArray()));}
