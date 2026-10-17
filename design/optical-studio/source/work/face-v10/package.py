import bpy,bmesh,numpy as np,json,shutil
from pathlib import Path
from mathutils import Matrix
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v10';stage=out/'release';(stage/'assembly').mkdir(parents=True,exist_ok=True);(stage/'print').mkdir(exist_ok=True)
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v10';manifest=json.load(open(r/'parts.json'));reports={};mapping={'face':'face-screen','head-upper':'head-upper-final','head-lower':'head-lower-final','magnet-carrier':'magnet-carrier-pockets'}
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for name,source in mapping.items():
 bpy.ops.wm.stl_import(filepath=str(out/(source+'.stl')));o=bpy.context.object;oldout=out;out=stage/'assembly';rep=export(o,name);out=oldout;print(name,rep,flush=True);assert rep['components']==1 and rep['nonManifoldEdges']==0
 if name=='magnet-carrier':M=Matrix(((0,1,0,0),(0,0,1,0),(1,0,0,0),(0,0,0,1)))
 else:
  mat=next(p['printTransform']for p in manifest['parts']if p['name']==name);M=Matrix([[mat[j*4+i]for j in range(4)]for i in range(4)])
 for v in o.data.vertices:v.co=M@v.co
 lo=np.min([v.co[:]for v in o.data.vertices],axis=0);shift=Matrix.Translation((0,0,-float(lo[2])));M=shift@M
 for v in o.data.vertices:v.co=shift@v.co
 bounds=np.array([v.co[:]for v in o.data.vertices]);dims=bounds.max(0)-bounds.min(0);oldout=out;out=stage/'print';export(o,name);out=oldout;rep.update({'dimensionsMM':dims.tolist(),'printTransform':[M[i][j]for j in range(4)for i in range(4)],'fitsAD5M':bool(max(dims)<=220),'fitsA1':bool(max(dims)<=256)});reports[name]=rep;bpy.data.objects.remove(o,do_unlink=True)
for i in range(4):
 for side in ['head','face']:
  name=f'magnet-clear-{i}-{side}-magnet';shutil.copy2(out/(name+'.stl'),stage/'assembly'/(name+'.stl'))
(out/'release-geometry.json').write_text(json.dumps(reports,indent=2))
