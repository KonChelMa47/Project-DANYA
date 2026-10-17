import bpy,numpy as np,time
from mathutils.bvhtree import BVHTree
from pathlib import Path
root=Path(__file__).resolve().parent.parent;t=time.time()
def log(s):print(f'[{time.time()-t:.1f}s] {s}',flush=True)
def coords(m,field='co'):
 a=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get(field,a);return a.reshape(-1,3)
bpy.ops.wm.open_mainfile(filepath=str(root/'work/candidate.blend'))
with bpy.data.libraries.load(str(root/'work/common_simplified.blend'),link=False) as (s,d):d.objects=['Unified_work_surface']
master=d.objects[0];m=master.data;m.calc_loop_triangles();sv=coords(m);sn=coords(m,'normal');sf=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',sf);sf=sf.reshape(-1,3)
tree=BVHTree.FromPolygons(sv.tolist(),sf.tolist(),all_triangles=True);log('Built common normal reference')
fit=np.load(root/'work/seam_fit.npz')
for name in ['DANYA-face-back','DANYA-face-shell']:
 m=bpy.data.objects[name].data;v=coords(m)
 matches=[tree.find_nearest(p) for p in v];idx=np.array([r[2] for r in matches]);q=np.array([r[0] for r in matches]);tri=sf[idx]
 a,b,c=sv[tri[:,0]],sv[tri[:,1]],sv[tri[:,2]];u=b-a;w=c-a;p=q-a
 uu=np.einsum('ij,ij->i',u,u);ww=np.einsum('ij,ij->i',w,w);uw=np.einsum('ij,ij->i',u,w);up=np.einsum('ij,ij->i',u,p);wp=np.einsum('ij,ij->i',w,p)
 det=np.maximum(uu*ww-uw*uw,1e-20);beta=(ww*up-uw*wp)/det;gamma=(uu*wp-uw*up)/det
 normals=sn[tri[:,0]]*(1-beta-gamma)[:,None]+sn[tri[:,1]]*beta[:,None]+sn[tri[:,2]]*gamma[:,None]
 normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
 li=np.empty(len(m.loops),dtype=np.int32);m.loops.foreach_get('vertex_index',li);ln=normals[li]
 yz=(v[:,1:]-[0,135])/100;fx=sum(c*yz[:,0]**i*yz[:,1]**j for c,(i,j) in zip(fit['coeff'],fit['powers']));on=np.abs(v[:,0]-fx)<.0001
 for poly in m.polygons:
  if all(on[i] for i in poly.vertices):ln[poly.loop_start:poly.loop_start+poly.loop_total]=poly.normal
 m.normals_split_custom_set(ln.tolist());m.update();log('Finished seamless outer normals: '+name)
bpy.data.objects.remove(master,do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/candidate.blend'));log('Saved shading finish')
