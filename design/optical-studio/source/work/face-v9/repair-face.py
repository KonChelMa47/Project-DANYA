import bpy,bmesh,json,numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(out/'face.stl'));o=bpy.context.object;o.data.remesh_voxel_size=.18;o.data.remesh_voxel_adaptivity=0;bpy.ops.object.voxel_remesh();print('remeshed',len(o.data.vertices),flush=True)
# Snap only facial exterior vertices back to the unchanged original facial surface.
a=np.load(r/'face-v8/screen-data.npz');v=a['before'];f=a['outerFaces'];tree=BVHTree.FromPolygons(v.tolist(),f.tolist());bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.normal_update();snaps=0
for p in bm.verts:
 h,n,_,d=tree.find_nearest(p.co)
 if d<.32 and n.dot(p.normal)>.6:p.co=h;snaps+=1
bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(o.data);bm.free();print('snaps',snaps,flush=True)
bpy.ops.wm.stl_export(filepath=str(out/'face-repaired.stl'),export_selected_objects=True)
bm=bmesh.new();bm.from_mesh(o.data);u=set(bm.verts);comps=[]
while u:
 st=[u.pop()];vv=set(st)
 while st:
  for e in st.pop().link_edges:
   for p in e.verts:
    if p in u:u.remove(p);vv.add(p);st.append(p)
 comps.append(len(vv))
report={'voxelSizeMM':.18,'exteriorSnappedVertices':snaps,'componentVertexCounts':comps,'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volumeMM3':bm.calc_volume(signed=False)};bm.free();(out/'repair-check.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
