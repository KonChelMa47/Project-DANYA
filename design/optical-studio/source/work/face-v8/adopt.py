import bpy,bmesh,json,shutil,numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'face-v8';backup=r/'archive-before-face-v8'
g=json.load(open(v/'geometry-check.json'));s=json.load(open(v/'surface-check.json'));o=json.load(open(v/'optical-check.json'))
assert g['face']['components']==1 and g['face']['nonManifoldEdges']==0
assert s['face']['nonAdjacentCrossingsOver01MM']==0 and s['originalOuterSurfaceMaxDeviationMM']<.001
assert not o['reinforcementContacts'] and not o['faceContactsWith002MMForwardSeamRelief'] and o['missingPaths']==0
backup.mkdir(exist_ok=True)
for p in ['assembly/face.stl','print/face.stl','parts.json','print-validation.json','projection-prototype.blend','projection-prototype-print.zip','ASSEMBLY.md','MAGNET_ASSEMBLY.md']:
 q=backup/p;q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():shutil.copy2(r/p,q)
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'));old=bpy.data.objects['face'];mats=list(old.data.materials);cols=list(old.users_collection);bpy.data.objects.remove(old,do_unlink=True)
shutil.copy2(v/'face.stl',r/'assembly/face.stl');bpy.ops.wm.stl_import(filepath=str(r/'assembly/face.stl'));obj=bpy.context.object;obj.name='face'
for mat in mats:obj.data.materials.append(mat)
for c in list(obj.users_collection):c.objects.unlink(obj)
for c in cols:c.objects.link(obj)
obj['revision']='face-v8';obj['note']='Inward0.5mm nominal offset; reinforced magnet arms. Read face-v8 optical report: edge shadows remain; no physical illumination claim.'
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
manifest=json.load(open(r/'parts.json'));p=next(p for p in manifest['parts']if p['name']=='face');mat=p['printTransform'];M=Matrix([[mat[j*4+i]for j in range(4)]for i in range(4)])
for vert in obj.data.vertices:vert.co=M@vert.co
lo=np.min([v.co[:]for v in obj.data.vertices],axis=0);hi=np.max([v.co[:]for v in obj.data.vertices],axis=0);assert abs(lo[2])<.001
bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj;bpy.ops.wm.stl_export(filepath=str(r/'print/face.stl'),export_selected_objects=True)
p['revision']='face-v8';p['notes']='Nominal inward0.5mm screen offset. Four magnet arms strengthened with0.8mm web and2mm edge ribs,20mm height. Same magnets, pockets and exterior seam. Edge shading quantified in face-v8/optical-check.json; physical strength/transmission untested.'
manifest['revision']='face-v8';(r/'parts.json').write_text(json.dumps(manifest,indent=2))
validation=json.load(open(r/'print-validation.json'));p=next(p for p in validation if p['name']=='face');p.update({'dimensionsMM':(hi-lo).tolist(),'nonManifoldEdges':0,'volumeMM3':g['face']['volumeMM3'],'fitsA1':True,'fitsAD5M':True,'assemblyToPrintMatrix':mat});(r/'print-validation.json').write_text(json.dumps(validation,indent=2));print('ADOPTED',flush=True)
