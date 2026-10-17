exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);o=load(out/'carrier-voxel.stl');bm=bmesh.new();bm.from_mesh(o.data);unseen=set(bm.verts);groups=[]
while unseen:
 comp={unseen.pop()};stack=list(comp)
 while stack:
  v=stack.pop()
  for e in v.link_edges:
   w=e.other_vert(v)
   if w in unseen:unseen.remove(w);comp.add(w);stack.append(w)
 groups.append(comp)
groups.sort(key=len,reverse=True);print('component vertex counts',[len(g)for g in groups],flush=True);bmesh.ops.delete(bm,geom=[v for g in groups[1:]for v in g],context='VERTS');bm.to_mesh(o.data);bm.free();m=o.modifiers.new('reduce planar detail','DECIMATE');m.ratio=.15;bpy.context.view_layer.objects.active=o;bpy.ops.object.modifier_apply(modifier=m.name);print('finished',export(o,'carrier-finished'),flush=True)
