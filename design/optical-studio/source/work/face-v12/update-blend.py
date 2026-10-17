import bpy,json
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'face-v12';audit=json.load(open(v/'adoption-audit.json'));names=[Path(p).stem for p in audit['changedSTLs']if p.startswith('assembly/')]
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
for name in names:
 old=bpy.data.objects.get(name);materials=list(old.data.materials)if old and old.type=='MESH' else [];cols=list(old.users_collection)if old else [bpy.context.scene.collection]
 if old:bpy.data.objects.remove(old,do_unlink=True)
 bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;o.name=name
 for mat in materials:o.data.materials.append(mat)
 for c in list(o.users_collection):c.objects.unlink(o)
 for c in cols:c.objects.link(o)
 o['revision']='face-v12';o['assembly_note']='Concealed 3x1mm upper magnets, smooth perimeter, no external handles. Face unit removes forward. Sampled removal and ray checks passed; physical force/strength untested.'
bpy.context.scene['DANYA_revision']='face-v12';bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'));print('updated',names,flush=True)
