import fs from 'node:fs';import {spawnSync} from 'node:child_process';
const names=['magnet-carrier','head-upper','head-lower',...Array.from({length:4},(_,i)=>['magnet-clear-'+i+'-face-magnet','magnet-clear-'+i+'-head-magnet']).flat()];
for(const name of names){let r=spawnSync(process.execPath,['work/face-v10/check-carrier.mjs',name],{encoding:'utf8'});if(r.status!==0)throw Error(r.stderr);console.log(r.stdout);}
