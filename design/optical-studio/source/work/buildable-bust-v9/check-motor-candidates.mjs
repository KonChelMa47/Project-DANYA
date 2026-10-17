import fs from 'node:fs';
import {STLLoader} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/loaders/STLLoader.js';
import {mergeGeometries} from '../../outputs/danya-simulator/node_modules/three/examples/jsm/utils/BufferGeometryUtils.js';
import {obs,check,stage} from '../buildable-bust-v8/route-check.mjs';
const root='outputs/danya-simulator/assets/buildable-bust-v9',data=JSON.parse(fs.readFileSync(root+'/motor-candidates.json')),rows=[];
for(const role of ['yaw','pitch','roll'])for(const xc of [18,22,26]){
 const tag=role+'-'+xc;
 const pp=data.parts.filter(p=>p.name.startsWith('printed-motor-'+tag+'-')||p.name.startsWith('hardware-'+tag+'-'));
 const geometries=pp.map(p=>{const b=fs.readFileSync(root+'/'+p.name+'.stl');return new STLLoader().parse(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));});
 const ignore=obs.filter(p=>p.name.startsWith('motor-'+role+'-')||p.name.startsWith('fastener-'+role+'-')||p.name==='frame-'+role+'-motor-support').map(p=>p.name);
 const result=check(mergeGeometries(geometries),stage[pp[0].parent],ignore,true);
 rows.push({role,xc,excludedReplacedParts:ignore,...result});console.log(tag,JSON.stringify(result));
}
fs.writeFileSync(root+'/motor-candidate-check.json',JSON.stringify({scope:'27 sampled poses; surface intersections against v8 retained geometry, official motor/projector CAD, shell and sampled chief rays. Old same-role saddle, fasteners and support removed; replacement frame connection not yet designed. Not a full assembly pass.',rows},null,2));
