import bpy,bmesh,json,numpy as np
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v9'
s=Path('work/projection-prototype/magnets-v2/build.py').read_text();exec(s[s.index('def clean'):s.index("bpy.ops.object.select_all(action='SELECT')")]);out=r/'face-v9'
def load(p):
 bpy.ops.wm.stl_import(filepath=str(p));return bpy.context.object
def boolean(a,b,op):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(op,'BOOLEAN');m.operation=op;m.solver='MANIFOLD';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);assert not a.modifiers;bpy.data.objects.remove(b,do_unlink=True)
def repair(o):
 bm=bmesh.new();bm.from_mesh(o.data)
 for _ in range(20):
  bad=[e for e in bm.edges if not e.is_manifold]
  if not bad:break
  remove={f for e in bad if e.is_boundary for f in e.link_faces if f.calc_area()<.01 and any(len(ee.link_faces)>2 for ee in f.edges)}
  if not remove:break
  bmesh.ops.delete(bm,geom=list(remove),context='FACES_ONLY');wire=[e for e in bm.edges if not e.link_faces]
  if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
 bm.to_mesh(o.data);bm.free()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
v=np.fromfile('work/face-v9/beam-exact.positions.bin','<f4').reshape(-1,3);f=np.fromfile('work/face-v9/beam-exact.indices.bin','<u4').reshape(-1,3);m=bpy.data.meshes.new('beam');m.from_pydata(v.tolist(),[],f.tolist());m.update();beam=bpy.data.objects.new('beam',m);bpy.context.collection.objects.link(beam)
collar=load(out/'collar-only.stl');repair(collar)
for o in [collar,beam]:
 bm=bmesh.new();bm.from_mesh(o.data);print(o.name,len(bm.verts),sum(not e.is_manifold for e in bm.edges),sum(f.calc_area()<1e-10 for f in bm.faces),flush=True);bm.free()
boolean(collar,beam,'DIFFERENCE');repair(collar);print('collar',export(collar,'collar-exact-clear'),flush=True)
face=load(r/'face-v8/screen-after.stl');boolean(face,collar,'UNION');repair(face);rep=export(face,'face-exact-fit');(out/'exact-fit-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
