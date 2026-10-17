import bpy,bmesh
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'face-v12';out.mkdir(exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for p in [r/'assembly/face.stl',r/'archive-before-face-v11/assembly/head-upper.stl',r/'archive-before-face-v11/assembly/head-lower.stl']:
 bpy.ops.wm.stl_import(filepath=str(p))
bpy.ops.object.select_all(action='SELECT');bpy.context.view_layer.objects.active=bpy.context.selected_objects[0];bpy.ops.object.join();o=bpy.context.object;bm=bmesh.new();bm.from_mesh(o.data);h=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False);bmesh.ops.delete(bm,geom=h['geom_interior']+h['geom_unused'],context='VERTS');bm.to_mesh(o.data);bm.free();bpy.ops.wm.stl_export(filepath=str(out/'head-envelope.stl'),export_selected_objects=True)
