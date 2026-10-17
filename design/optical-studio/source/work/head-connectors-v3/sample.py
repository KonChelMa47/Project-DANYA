import bpy,bmesh,numpy as np,json,math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from pathlib import Path
r=Path.cwd();bpy.ops.wm.stl_import(filepath=str(r/'outputs/danya-simulator/assets/projection-prototype/assembly/head.stl'));o=bpy.context.object;b=bmesh.new();b.from_mesh(o.data);bmesh.ops.remove_doubles(b,verts=list(b.verts),dist=1e-5);b.to_mesh(o.data);b.free();tree=BVHTree.FromPolygons([v.co for v in o.data.vertices],[p.vertices[:]for p in o.data.polygons]);rows=[]
for z in [50.7,52,54,56,58,60,64,68,72,76]:
 vals=[]
 for j in range(180):
  t=j*math.tau/180;p=Vector((-22.0161686,.3457451,z));d=Vector((math.cos(t),math.sin(t),0));hits=[]
  for it in range(20):
   h,_,_,dist=tree.ray_cast(p,d,150)
   if h is None:break
   hits.append(math.hypot(h.x+22.0161686,h.y-.3457451));p=h+d*.0001
  vals.append([min(hits)if hits else None,max(hits)if hits else None])
 a=np.array([[x,y]for x,y in vals if x is not None]);print(z,'hits',len(a),'inner/outer span',a.min(0).tolist(),a.max(0).tolist(),'wall',np.quantile(a[:,1]-a[:,0],[0,.05,.5,.95,1]).tolist(),flush=True);rows.append({'z':z,'radii':vals})
(r/'work/head-connectors-v3/neck-sections.json').write_text(json.dumps(rows))
