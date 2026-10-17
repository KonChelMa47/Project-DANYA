from pathlib import Path
import numpy as np,json
p=Path(__file__).resolve().parent;r=p.parents[1]/'outputs/danya-simulator/assets';meshes={};verts=[];faces=[];off=0
for name in ['head','face']:
 v=np.fromfile(r/f'{name}.positions.bin','<f4').reshape(-1,3);f=np.fromfile(r/f'{name}.indices.bin','<u4').reshape(-1,3);meshes[name]=(v,f);verts.append(v);faces.append(f+off);off+=len(v)
globalv,inv=np.unique(np.concatenate(verts),axis=0,return_inverse=True);gf=inv[np.concatenate(faces)];keys=np.sort(gf,axis=1);_,ii,cc=np.unique(keys,axis=0,return_inverse=True,return_counts=True);cap=cc[ii]>1;start=0
for name in ['head','face']:
 v,f=meshes[name];mask=cap[start:start+len(f)];start+=len(f);np.savez(p/f'{name}-original.npz',v=v,f=f,cap=mask);keep=f[~mask];parent=np.arange(len(v))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for tri in keep:
  a=find(tri[0])
  for b in tri[1:]:
   b=find(b)
   if a!=b:parent[b]=a
 roots=np.array([find(i)for i in range(len(v))]);comps=[]
 for root in np.unique(roots):
  vert=np.where(roots==root)[0];tri=keep[roots[keep[:,0]]==root]
  if len(tri):comps.append((len(vert),float(v[vert,0].mean()),root,len(tri)))
 print(name,'caps',int(mask.sum()),'components',sorted(comps,reverse=True)[:10],flush=True)
 if name=='face':
  comps=sorted(comps,reverse=True)[:2];outer=max(comps,key=lambda x:float(v[roots==x[2],0].max()))[2];of=keep[roots[keep[:,0]]==outer];of.astype('<u4').tofile(p/'original-face-outer.indices.bin');v.astype('<f4').tofile(p/'original-face.positions.bin')
