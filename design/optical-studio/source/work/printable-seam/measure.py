import bpy,numpy as np,json
from pathlib import Path
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'printable-seam';out.mkdir(exist_ok=True);bpy.ops.wm.read_factory_settings(use_empty=True);rep={}
for name,p in [('magnet-carrier',r/'assembly/magnet-carrier.stl'),('face',r/'assembly/face.stl'),('head-upper',r/'head-split-quarter/head-upper.stl'),('head-lower',r/'head-split-quarter/head-lower.stl')]:
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;tree=BVHTree.FromPolygons([v.co for v in o.data.vertices],[p.vertices[:]for p in o.data.polygons]);rows=[]
 for f in list(o.data.polygons)[::5]:
  center=sum((o.data.vertices[i].co for i in f.vertices),bpy.mathutils.Vector())/len(f.vertices) if False else f.center.copy();n=f.normal.copy();h=tree.ray_cast(center-n*.01,-n,5)
  if h[0] is not None and h[3]>.005:rows.append([*center,float(h[3]+.01),float(f.area)])
 arr=np.array(rows);np.save(out/(name+'-thickness.npy'),arr);thin=arr[arr[:,3]<.8];rep[name]={'samples':len(arr),'thinBelow08':len(thin),'minimumMM':float(arr[:,3].min()),'quantiles':np.quantile(arr[:,3],[.01,.1,.5]).tolist(),'thinBounds':[[float(thin[:,k].min()),float(thin[:,k].max())]for k in range(3)]if len(thin)else[]};print(name,rep[name],flush=True);bpy.data.objects.remove(o,do_unlink=True)
(out/'before-thickness.json').write_text(json.dumps(rep,indent=2))
