exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
name='magnet-carrier-covered';d=json.load(open(out/(name+'-surface.json')));o=load(out/(name+'.stl'));clean(o);bm=bmesh.new();bm.from_mesh(o.data)
keys={tuple(round(float(c),4)for c in p)for e in d['examples']for k in ['a','b']for p in e[k]};vs={v for v in bm.verts if tuple(round(c,4)for c in v.co)in keys};bad={f for v in vs for f in v.link_faces};print('removing',len(bad),flush=True);bmesh.ops.delete(bm,geom=list(bad),context='FACES_ONLY')
wire=[e for e in bm.edges if not e.link_faces]
if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
boundary=[e for e in bm.edges if e.is_boundary];print('boundary',len(boundary),flush=True);ret=bmesh.ops.holes_fill(bm,edges=boundary,sides=0);print('filled',len(ret['faces']),flush=True)
bmesh.ops.triangulate(bm,faces=list(bm.faces));bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(o.data);bm.free();print(export(o,name),flush=True)
