exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
print('SOLVERS',[x.identifier for x in bpy.types.BooleanModifier.bl_rna.properties['solver'].enum_items],flush=True)
v=src/'split-head-v6';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def inspect(o,label):
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);print(label,'verts',len(bm.verts),'bad',sum(not e.is_manifold for e in bm.edges),'vol',bm.calc_volume(),flush=True);bm.free()
bpy.ops.wm.stl_import(filepath=str(src/'archive-before-printable-mounts-v5/assembly/head.stl'));h=bpy.context.object;clean(h);inspect(h,'ORIGINAL')
for angle in [103,123,240,257]:
 bpy.ops.wm.stl_import(filepath=str(v/f'flange-{angle}.stl'));f=bpy.context.object;clean(f);inspect(f,'FLANGE'+str(angle));boolean(h,f,'UNION');clean(h);inspect(h,'UNION'+str(angle))
