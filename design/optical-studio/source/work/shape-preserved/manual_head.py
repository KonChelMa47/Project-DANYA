import numpy as np,bpy,bmesh
from pathlib import Path
p=Path(__file__).resolve().parent;data=np.load(p/'head-original.npz');v=data['v'].copy();f=data['f'];keep=f[~data['cap']];face=np.fromfile(p/'face.positions.bin','<f4').reshape(-1,3);nn=len(face)//2;coords={tuple(x):i for i,x in enumerate(face[:nn])}
edges=np.concatenate([keep[:,[0,1]],keep[:,[1,2]],keep[:,[2,0]]]);_,ii,cc=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);be=edges[ii[cc==1]];adj={}
for a,b in be:adj.setdefault(int(a),[]).append(int(b));adj.setdefault(int(b),[]).append(int(a))
assert all(len(x)==2 for x in adj.values());loops=[];seen=set()
for a in adj:
 if a in seen:continue
 loop=[];cur=a;last=-1
 while cur not in seen:seen.add(cur);loop.append(cur);nxt=adj[cur][0] if adj[cur][0]!=last else adj[cur][1];last,cur=cur,nxt
 loops.append(np.array(loop))
assert len(loops)==2;loops.sort(key=lambda a:sum(tuple(x)in coords for x in v[a]),reverse=True);outer,inner=loops;print('loops',len(outer),len(inner),'matches',sum(tuple(x)in coords for x in v[outer]),flush=True)
ang=lambda a:np.arctan2(a[:,2]-135,a[:,1]);order=np.argsort(ang(v[outer]));outer=outer[order];ta=ang(v[outer]);fi=np.array([coords[tuple(x)] for x in v[outer]]);targets=face[fi+nn].copy();radial=targets.copy();radial[:,0]=0;radial[:,2]-=135;radial/=np.linalg.norm(radial,axis=1)[:,None];targets+=radial*.35
angles=ang(v[inner]);extended=np.concatenate([ta[-1:]-2*np.pi,ta,ta[:1]+2*np.pi]);values=np.concatenate([targets[-1:],targets,targets[:1]])
for axis in range(3):v[inner,axis]=np.interp(angles,extended,values[:,axis])
a=outer;b=inner[np.argsort(ang(v[inner]))];ta=ang(v[a]);tb=ang(v[b]);i=j=0;tri=[]
while i<len(a)or j<len(b):
 an=ta[(i+1)%len(a)]+(2*np.pi if i+1>=len(a)else 0);bn=tb[(j+1)%len(b)]+(2*np.pi if j+1>=len(b)else 0)
 if j>=len(b)or(i<len(a)and an<bn):tri.append([a[i%len(a)],a[(i+1)%len(a)],b[j%len(b)]]);i+=1
 else:tri.append([a[i%len(a)],b[(j+1)%len(b)],b[j%len(b)]]);j+=1
f=np.concatenate([keep,np.array(tri)]);m=bpy.data.meshes.new('head internal joint');m.from_pydata(v.tolist(),[],f.tolist());m.update();bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));print('nonmanifold',sum(not e.is_manifold for e in bm.edges),'volume',bm.calc_volume(signed=True),flush=True);bm.to_mesh(m);bm.free();m.calc_loop_triangles();f=np.empty(len(m.loop_triangles)*3,np.uint32);m.loop_triangles.foreach_get('vertices',f);v.astype('<f4').tofile(p/'manual-head.positions.bin');f.tofile(p/'manual-head.indices.bin')
