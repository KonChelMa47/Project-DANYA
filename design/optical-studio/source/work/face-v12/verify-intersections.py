import json,numpy as np
from pathlib import Path
for name in ['magnet-carrier-covered','head-upper-covered']:
 d=json.load(open(Path('outputs/danya-simulator/assets/projection-prototype/face-v12')/(name+'-surface.json')));rows=[]
 for e in d['examples']:
  if e['type']!='crossing':continue
  a,b=np.array(e['a']),np.array(e['b']);n=np.cross(a[1]-a[0],a[2]-a[0]);n/=np.linalg.norm(n);m=np.cross(b[1]-b[0],b[2]-b[0]);m/=np.linalg.norm(m);direction=np.cross(n,m);direction/=np.linalg.norm(direction)
  def section(v,n,p):
   ds=(v-p)@n;pts=[]
   for i in range(3):
    j=(i+1)%3
    if abs(ds[i])<1e-9:pts.append(v[i]@direction)
    if ds[i]*ds[j]<0:pts.append((v[i]+(v[j]-v[i])*ds[i]/(ds[i]-ds[j]))@direction)
   return (min(pts),max(pts))if pts else(None,None)
  aa=section(a,m,b[0]);bb=section(b,n,a[0]);length=max(0,min(aa[1],bb[1])-max(aa[0],bb[0]))if aa[0]is not None and bb[0]is not None else 0
  rows.append((e['length'],length,a.mean(0).tolist()))
 print(name,'overlap>0.01',sum(r[1]>.01 for r in rows),'of',len(rows));print(sorted(rows,key=lambda r:r[1],reverse=True)[:10])
