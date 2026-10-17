import fs from 'node:fs';
import * as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {obs} from './baseline-obs.mjs';
import {around} from './motion.mjs';
const layout=JSON.parse(fs.readFileSync('work/torso-flat/torso-v3-layout.json'));
for(const name of ['mirror','foldMirror']){
 const o=layout[name],s=new T.Shape((o.outline||[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).map(([x,y])=>new T.Vector2(x*o.width,y*o.height)));
 const g=new T.ExtrudeGeometry(s,{depth:1.1,bevelEnabled:false,steps:1}).translate(0,0,-1.1).applyMatrix4(new T.Matrix4().compose(new T.Vector3(...o.position),new T.Quaternion().setFromEuler(new T.Euler(...o.rotation.map(v=>v*Math.PI/180),'XYZ')),new T.Vector3(1,1,1)));g.boundsTree=new MeshBVH(g);obs.push({name:name+'-glass',stage:3,g});
}
const targets=obs.filter(o=>o.name.startsWith('motor-')&&!o.name.includes('saddle')&&!o.name.includes('cap')||o.name.endsWith('-glass'));
const frames=obs.filter(o=>o.name.startsWith('frame-'));
const hits=[];
for(const a of targets){
 for(const b of frames){let first=null,seen=new Set();
  outer:for(let y=-30;y<=30;y+=3)for(let p=0;p<=12;p+=3)for(let r=-6;r<=6;r+=3){
    const angles={y,p,r},ms=[new T.Matrix4(),around({y,p:0,r:0}),around({y,p,r:0}),around(angles)],m=ms[a.stage].clone().invert().multiply(ms[b.stage]),key=m.elements.map(x=>x.toFixed(7)).join();if(seen.has(key))continue;seen.add(key);
    if(a.g.boundsTree.intersectsGeometry(b.g,m)){first={y,p,r};break outer;}
  }
  if(first){hits.push({a:a.name,b:b.name,first});console.log(JSON.stringify(hits.at(-1)));}
 }
}
fs.writeFileSync('work/buildable-bust-v10/visible-collisions.json',JSON.stringify({scope:'v8 visible assembly. Three-degree sample within yaw +/-30,pitch0..12,roll+/-6. Glass included. Intended mounting contacts require separate interpretation.',hits},null,2));
