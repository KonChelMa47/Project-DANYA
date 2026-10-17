import fs from'node:fs';import{obs}from'./full-route-check.mjs';
const cx=-22.01616859436035,cy=.3457450866699219,rad=Math.PI/180,pmax=12*rad,rmax=6*rad,normalize=t=>Math.atan2(Math.sin(t),Math.cos(t));let rows=[];
for(const o of obs.filter(o=>o.name==='projector'||o.name.startsWith('projector-tray-t5'))){let best={z:Infinity};const a=o.g.attributes.position;
 for(let i=0;i<a.count;i++){const X=a.getX(i)-cx,B=a.getY(i)-cy,A=a.getZ(i)-20;const rs=[-rmax,rmax],rr=normalize(Math.atan2(B,A)+Math.PI);if(rr>=-rmax&&rr<=rmax)rs.push(rr);
  for(const r of rs){const C=24+A*Math.cos(r)+B*Math.sin(r),ps=[0,pmax],pp=normalize(Math.atan2(-X,C)+Math.PI);if(pp>=0&&pp<=pmax)ps.push(pp);
   for(const p of ps){const z=-4-X*Math.sin(p)+C*Math.cos(p);if(z<best.z)best={z,pitch:p/rad,roll:r/rad,vertex:i};}
  }
 }
 rows.push({name:o.name,...best});
}
rows.sort((a,b)=>a.z-b.z);console.log(JSON.stringify(rows.slice(0,6),null,2));fs.writeFileSync('outputs/danya-simulator/assets/buildable-bust-v10/projector-bottom-envelope.json',JSON.stringify({scope:'Exact minimumZ of mesh vertices over continuouspitch0..12 androll+-6; yawdoesnotchangeZ. Endpoint+stationaryangle extrema. NominalrigidCADonly.',rows},null,2));
