import bpy,bmesh
p='outputs/danya-simulator/assets/projection-prototype/magnets-v2/integrated-1-head-addition.stl';bpy.ops.wm.stl_import(filepath=p);b=bmesh.new();b.from_mesh(bpy.context.object.data);bmesh.ops.remove_doubles(b,verts=list(b.verts),dist=1e-5)
print('BAD',[(len(e.link_faces),e.calc_length(),[list(v.co)for v in e.verts])for e in b.edges if not e.is_manifold])
