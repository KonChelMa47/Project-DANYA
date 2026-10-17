import bpy,bmesh
from pathlib import Path
bpy.ops.wm.read_factory_settings(use_empty=True);r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype'
for p in [r/'thick-head-v3/thick-head-clean.stl']+[r/'split-head-v6'/f'flange-{a}.stl'for a in [103,123,240,257]]:
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);print(p.name,'nonmanifold',sum(not e.is_manifold for e in bm.edges),'degen',sum(f.calc_area()<1e-9 for f in bm.faces),flush=True);bm.free();bpy.data.objects.remove(o,do_unlink=True)
