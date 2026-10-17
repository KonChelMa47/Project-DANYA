exec(open('work/face-v9/fit-collar-screen.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
v=np.fromfile('work/face-v9/beam-outer.positions.bin','<f4').reshape(-1,3);f=np.fromfile('work/face-v9/beam-outer.indices.bin','<u4').reshape(-1,3);m=bpy.data.meshes.new('outer cone');m.from_pydata(v.tolist(),[],f.tolist());m.update();beam=bpy.data.objects.new('outer cone',m);bpy.context.collection.objects.link(beam);screen=load(r/'face-v8/screen-after.stl');boolean(beam,screen,'DIFFERENCE');repair(beam)
bm=bmesh.new();bm.from_mesh(beam.data);u=set(bm.verts);groups=[]
while u:
 st=[u.pop()];vs=set(st)
 while st:
  for e in st.pop().link_edges:
   for p in e.verts:
    if p in u:u.remove(p);vs.add(p);st.append(p)
 groups.append(vs)
for vs in groups:
 lo=[min(p.co[i]for p in vs)for i in range(3)];hi=[max(p.co[i]for p in vs)for i in range(3)];print('cavity component',len(vs),lo,hi,flush=True)
 if lo[0]>-200:bmesh.ops.delete(bm,geom=list(vs),context='VERTS')
bm.to_mesh(beam.data);bm.free();rep=export(beam,'light-cavity');print(rep,flush=True)
