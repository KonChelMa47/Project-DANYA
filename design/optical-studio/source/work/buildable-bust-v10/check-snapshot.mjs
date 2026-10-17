import fs from 'node:fs';
import crypto from 'node:crypto';
const root='outputs/danya-simulator/assets/buildable-bust-v10';
const files=fs.readdirSync(root).filter(n=>/\.(stl|bin)$/.test(n)||['trial-structural-candidate.json','trial-drive-layout.json','transmission.json'].includes(n)).map(n=>root+'/'+n);
files.push('work/torso-flat/torso-v3-layout.json','work/buildable-bust-v10/lift-route-check.mjs','work/buildable-bust-v10/lift-motion.mjs');
const hash=()=>Object.fromEntries(files.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')]));
const before=hash(),started=new Date().toISOString();
const {obs,check}=await import('./lift-route-check.mjs');
const wanted=['trial-rear-chassis','trial-floor','trial-pitch-support','trial-frame-projector-right','trial-frame-optics-left','trial-left-head-rail','trial-right-head-rail','trial-roll-support','mirror-head-cradle','motor-yaw','motor-pitch','motor-roll','projector','mirror-glass','foldMirror-glass','pitch-belt','roll-belt'];
const rows=[];
for(const name of wanted){const a=obs.find(o=>o.name===name);if(!a)throw Error('Missing '+name);const row={name,...check(a.g,a.stage,[name],false)};rows.push(row);console.log(JSON.stringify(row));}
const after=hash(),changed=files.filter(p=>before[p]!==after[p]);
fs.writeFileSync(root+'/snapshot-screen.json',JSON.stringify({started,finished:new Date().toISOString(),inputHashes:before,changedDuringRun:changed,validSnapshot:changed.length===0,scope:'525 combined poses: yaw -30..30, pitch 0..12, roll -6..6, 3 degree spacing. Surface intersections only; mating contacts reported. NOT continuous, containment, strength or optical certification.',rows},null,2));
if(changed.length)throw Error('Inputs changed during screen: '+changed.join(','));
