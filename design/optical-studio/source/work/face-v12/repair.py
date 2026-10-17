exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
o=load(out/'magnet-carrier.stl');clean(o)
bm=bmesh.new();bm.from_mesh(o.data)
for _ in range(20):
 bad=[e for e in bm.edges if not e.is_manifold]
 print('bad',len(bad),[(len(e.link_faces),[v.co[:]for v in e.verts])for e in bad][:10],flush=True)
 if not bad:break
 remove={f for e in bad if e.is_boundary for f in e.link_faces if f.calc_area()<.01 and any(len(ee.link_faces)>2 for ee in f.edges)}
 if not remove:break
 bmesh.ops.delete(bm,geom=list(remove),context='FACES_ONLY');wire=[e for e in bm.edges if not e.link_faces]
 if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
bm.to_mesh(o.data);bm.free();print(export(o,'magnet-carrier'),flush=True)
