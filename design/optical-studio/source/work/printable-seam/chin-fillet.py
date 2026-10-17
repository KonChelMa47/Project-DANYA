exec(open('work/printable-seam/restore-seat.py').read().split("bpy.ops.wm.stl_import(filepath=str(out/'head-upper.stl'))")[0])
from mathutils import Vector
bpy.ops.wm.stl_import(filepath=str(out/'head-lower.stl'));head=bpy.context.object
# A short outward rounded fillet at the fragile chin edge; retain the light-facing surface and split plane.
bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,location=(23.45,-30.90,50.60));o=bpy.context.object;o.scale=(1.2,3.0,1.2);bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);op(head,o,'UNION')
bpy.ops.wm.stl_import(filepath=str(out/'thin-lip-removal-tool.stl'));op(head,bpy.context.object,'DIFFERENCE');print(export(head,'head-lower'))
