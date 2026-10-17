import bpy,json,numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';trees=[]
for file in ['archive-before-integrated-magnets/assembly/head.stl','face-v8/screen-after.stl']:
 bpy.ops.wm.stl_import(filepath=str(r/file));o=bpy.context.object;trees.append(BVHTree.FromPolygons([v.co for v in o.data.vertices],[p.vertices[:]for p in o.data.polygons]))
s=np.load('work/buildable-bust-v10/seam.npy');s[:,2]+=8;rows=[]
for upper in [True,False]:
 for y in [-40,-25,0,25,40]:
  a=s[(abs(s[:,1]-y)<1.0)&((s[:,2]>200)if upper else(s[:,2]<110))];q=a[a[:,2].argmax()if upper else a[:,2].argmin()];sg=1 if upper else-1
  for dx in [-25,-15,-8,-3,-1,0,1,3]:
   x=float(q[0])+dx;hs=[]
   for tree in trees:
    p=Vector((x,y,150));hits=[]
    for _ in range(4):
     h=tree.ray_cast(p,Vector((0,0,sg)),200)[0]
     if h is None:break
     hits.append(h.z);p=h+Vector((0,0,sg*.001))
    hs.append(hits)
   rows.append({'top':upper,'y':y,'seam':q.tolist(),'dx':dx,'head':hs[0],'face':hs[1]})
Path('work/face-v9/probe.json').write_text(json.dumps(rows,indent=2))
