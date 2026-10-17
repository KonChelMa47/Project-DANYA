import fs from 'node:fs';import *as T from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';import {STLLoader} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/loaders/STLLoader.js';import {mergeGeometries} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/utils/BufferGeometryUtils.js';import {obs,check,stage} from '../buildable-bust-v8/route-check.mjs';
const root='outputs/danya-simulator/assets/buildable-bust-v9',parts=JSON.parse(fs.readFileSync(root+'/motor-candidates.json')).parts,drive=JSON.parse(fs.readFileSync('outputs/danya-simulator/assets/buildable-bust-v8/drive-layout.json')),rows=[];
function read(p){let b=fs.readFileSync(root+'/'+p.name+'.stl');return new STLLoader().parse(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
for(const [i,d]of drive.motors.entries()){if(i!==0)continue;
 const role=d.role,old=new T.Matrix4().compose(new T.Vector3(...d.tip),new T.Quaternion(...d.quaternion),new T.Vector3(1,1,1));
 const ignore=obs.filter(o=>o.name.startsWith('frame-')||o.name.startsWith('motor-'+role+'-')||o.name.startsWith('fastener-'+role+'-')||o.name==='motor-'+role||(role==='yaw'&&o.name==='horn-clutch-hub')).map(o=>o.name);
 for(const xc of [18]){
  const pp=parts.filter(p=>p.name.startsWith('printed-motor-'+role+'-'+xc+'-')||p.name.startsWith('hardware-'+role+'-'+xc+'-'));
  const geometries=pp.map(read);let motor=obs.find(o=>o.name==='motor-'+role).g.clone();for(const key of Object.keys(motor.attributes))if(!['position','normal'].includes(key))motor.deleteAttribute(key);if(motor.index)motor=motor.toNonIndexed();geometries.push(motor);
  const g=mergeGeometries(geometries);
  for(const dz of [5,10,15,20])for(const deg of [-180,-165,-150,-135,0]){let q=new T.Quaternion(...d.quaternion).multiply(new T.Quaternion().setFromAxisAngle(new T.Vector3(0,1,0),deg*Math.PI/180));let delta=new T.Matrix4().compose(new T.Vector3(d.tip[0],d.tip[1],d.tip[2]-dz),q,new T.Vector3(1,1,1)).multiply(old.clone().invert());let r=check(g.clone().applyMatrix4(delta),i,ignore,true);rows.push({role,xc,deg,dz,q:q.toArray(),...r});if(!r.hits.length&&!r.lightHits)console.log('PASS',role,xc,deg,dz);}
  console.log('done',role,xc);
 }
 fs.writeFileSync(root+'/yaw-lowered-search.json',JSON.stringify({scope:'27poses, v8 fabricated frames omitted for redesign; existing other motors kept. All clamp bolts/washers/nuts included. Surface contact and sampled chief rays only.',rows},null,2));
}
