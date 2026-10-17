import sys,json,shutil
from pathlib import Path
import numpy as np,pymeshfix
r=Path('outputs/danya-simulator/assets/projection-prototype/face-v12');name=sys.argv[1];p=r/(name+'.stl');d=json.load(open(r/(name+'-surface.json')))
raw=np.fromfile(p,dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3)
def key(a):return tuple(sorted(tuple(float(c)for c in q)for q in a))
lookup={key(a):i for i,a in enumerate(raw)};bad=set()
for e in d['examples']:
 for k in ['a','b']:
  idx=lookup.get(key(e[k]));assert idx is not None,(name,k)
  bad.add(idx)
verts=np.unique(f[list(bad)]);mask=np.isin(f,verts).any(1);print(name,'remove',mask.sum(),'faces from',len(f),flush=True);assert mask.sum()<len(f)*.03
backup=r/(name+'-before-local-repair.stl')
if not backup.exists():shutil.copy2(p,backup)
m=pymeshfix.PyTMesh(False);m.load_array(v.astype(float),f[~mask].astype(np.int32));print('holes filled',m.fill_small_boundaries(refine=False),flush=True);m.clean(max_iters=3,inner_loops=1);w,g=m.return_arrays();tri=w[g];o=np.zeros(len(g),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);o['v']=tri;n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30);o['n']=n
with p.open('wb')as h:h.write(b'DANYA local intersection patch'.ljust(80,b'\0'));h.write(np.asarray([len(g)],dtype='<u4').tobytes());h.write(o.tobytes())
print('written',len(g),flush=True)
