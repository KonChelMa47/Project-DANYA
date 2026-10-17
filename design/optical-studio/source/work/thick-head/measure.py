import bpy,numpy as np,json
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype/thick-head-v3';bpy.ops.wm.read_factory_settings(use_empty=True);allv=[];allf=[]
for n in ['head-upper','head-lower']:
 bpy.ops.wm.stl_import(filepath=str(r/(n+'.stl')));o=bpy.context.object;k=len(allv);allv.extend(v.co.copy()for v in o.data.vertices);allf.extend(tuple(k+i for i in p.vertices)for p in o.data.polygons)
tree=BVHTree.FromPolygons(allv,allf);d=np.load(r/'thickness-samples.npz');rows=[]
for i in range(0,len(d['v']),12):
 p=Vector(d['v'][i]);n=Vector(d['norm'][i]);start=p+n*10;h=tree.ray_cast(start,-n,30)[0]
 if h is None:continue
 inner=tree.ray_cast(h-n*.015,-n,25)[0]
 if inner is None:continue
 thickness=(inner-h).length;rows.append([*p,thickness,float(d['before'][i])])
a=np.array(rows);main=a[(a[:,0]<0)&(a[:,2]>65)&(a[:,2]<260)&(abs(a[:,2]-165)>2)];report={'method':'At every12th original exterior vertex, cast along its smoothed exterior normal through the final assembled head. First two surface hits define local wall. Main body subset:X<0,Z65..260, excluding±2mm aroundsplit. Finite samples, not a global minimum proof.','samples':len(a),'mainBodySamples':len(main),'mainBeforePercentiles':np.percentile(main[:,4],[0,1,10,50,90,100]).tolist(),'mainAfterPercentiles':np.percentile(main[:,3],[0,1,10,50,90,100]).tolist(),'mainBelow38MM':int((main[:,3]<3.8).sum()),'allAfterPercentiles':np.percentile(a[:,3],[0,1,10,50,90,100]).tolist()};np.save(r/'final-wall-probes.npy',a);(r/'wall-thickness-check.json').write_text(json.dumps(report,indent=2));print(report)
