import bpy,bmesh,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parent.parent
bpy.ops.wm.open_mainfile(filepath=str(root/'work/candidate.blend'))
d=np.load(root/'work/DANYA-face-shell-repaired.npz');v=d['v'];f=d['f'];area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
pairs=[]
for tri in v[f[area<1e-10]]:
 es=[(tri[i],tri[j]) for i,j in [(0,1),(1,2),(2,0)]];a,b=min(es,key=lambda x:np.linalg.norm(x[0]-x[1]));pairs.append((tuple(a),tuple(b)))
for name in ['DANYA-face-back','DANYA-face-shell']:
 m=bpy.data.objects[name].data;bm=bmesh.new();bm.from_mesh(m)
 byco={tuple(v.co):v for v in bm.verts};mapping={byco[a]:byco[b] for a,b in pairs if a in byco and b in byco}
 if mapping:bmesh.ops.weld_verts(bm,targetmap=mapping)
 bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(m);bm.free();m.update();m.calc_loop_triangles()
 v=np.empty(len(m.vertices)*3,dtype=np.float32);m.vertices.foreach_get('co',v);f=np.empty(len(m.loop_triangles)*3,dtype=np.int32);m.loop_triangles.foreach_get('vertices',f)
 np.savez(root/'work'/f'{name}-repaired.npz',v=v.reshape(-1,3),f=f.reshape(-1,3))
 print(name,'collapsed',len(mapping),'sub-tolerance sliver edge(s)',flush=True)
bpy.ops.wm.save_as_mainfile(filepath=str(root/'work/candidate.blend'))
