import fs from 'node:fs';
import crypto from 'node:crypto';
const root='outputs/danya-simulator/assets/buildable-bust-v10';
const files=fs.readdirSync(root).filter(n=>/\.(stl|bin)$/.test(n)||['revised-structural-candidate.json','revised-drive-layout.json','revised-transmission.json'].includes(n)).map(n=>root+'/'+n);
files.push('work/torso-flat/torso-v3-layout.json','work/buildable-bust-v10/revised-complete-route-check.mjs','work/buildable-bust-v10/lift-motion.mjs');
files.push('outputs/danya-simulator/assets/torso-flat-study/face.positions.bin','outputs/danya-simulator/assets/torso-flat-study/face.indices.bin','work/internal-projector/cad-mesh/kit.local.positions.bin','work/internal-projector/cad-mesh/kit.indices.bin','outputs/danya-simulator/assets/buildable-bust-v2/st3215-official.stl','outputs/danya-simulator/assets/buildable-bust-v5/gear-layout.json','outputs/danya-simulator/src/optics.js','work/buildable-bust-v4/motion.mjs','outputs/danya-simulator/node_modules/three/package.json','outputs/danya-simulator/node_modules/three-mesh-bvh/package.json');
const hash=()=>Object.fromEntries(files.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')]));
const before=hash(),started=new Date().toISOString();
const {obs,check}=await import('./revised-complete-route-check.mjs');
const wanted=obs.filter(o=>o.name.startsWith('magnet-clear-')||['body','head','face','drop-projector-tray-t5','lower-thrust-liner'].includes(o.name)).map(o=>o.name);
const rows=[];
for(const name of wanted){const a=obs.find(o=>o.name===name);if(!a)throw Error('Missing '+name);const row={name,...check(a.g,a.stage,[name],false)};rows.push(row);console.log(JSON.stringify(row));}
const after=hash(),changed=files.filter(p=>before[p]!==after[p]);
fs.writeFileSync(root+'/revised-shell-magnet-screen.json',JSON.stringify({runtime:process.version,started,finished:new Date().toISOString(),inputHashes:before,changedDuringRun:changed,validSnapshot:changed.length===0,scope:'Revised candidate:drop6/floor4.5, roll housing20deg/new300mm belt/upper-chest alternative. Lowered floor and body nut bosses modeled:525 combined poses: yaw -30..30, pitch 0..12, roll -6..6, 3 degree spacing. Surface intersections only; pinion parts use full-rotation conservative envelopes; mating contacts reported. NOT continuous, containment, strength or optical certification.',rows},null,2));
if(changed.length)throw Error('Inputs changed during screen: '+changed.join(','));
