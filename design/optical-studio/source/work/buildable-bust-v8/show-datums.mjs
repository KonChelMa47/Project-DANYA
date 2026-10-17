import fs from'node:fs';let d=JSON.parse(fs.readFileSync('work/buildable-bust-v8/interfaces.json'));for(let a of d)console.log(a.name,a.point,a.holes);
