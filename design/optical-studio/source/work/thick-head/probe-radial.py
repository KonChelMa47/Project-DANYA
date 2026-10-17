import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/thick-head-v3';bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.wm.stl_import(filepath=str(r.parent/'solid-seam-v2/head-lower.stl'));o=bpy.context.object;tree=BVHTree.FromPolygons([v.co for v in o.data.vertices],[p.vertices[:]for p in o.data.polygons]);d=np.load(r/'thickness-samples.npz')
for near in [(-27.6663,58.3273,66.6113),(-4.2639,-56.86,74.1159)]:
 i=np.argmin(np.linalg.norm(d['v']-near,axis=1));p=Vector(d['v'][i]);n=Vector(d['norm'][i]);print('P',p[:],'NORMAL',n[:]);n=-(p-Vector((-22,0,p.z))).normalized();start=Vector((-22,0,p.z));origin=start.copy();hits=[]
 for _ in range(8):
  h,nn,_,_=tree.ray_cast(start,-n,220)
  if h is None:break
  hits.append(((h-origin).length,h[:],nn[:]));start=h-n*.02
 print('HITS',hits)
