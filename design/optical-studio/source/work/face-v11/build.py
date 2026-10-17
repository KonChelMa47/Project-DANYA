exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=r/'face-v11';out.mkdir(exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
o=load(r/'assembly/magnet-carrier.stl')
# Two rigid underside finger tabs. Loads enter the continuous carrier, not the screen.
for y in [-18,18]:
 profile=[(31,56),(34,56),(34,50),(42,43),(42,39),(33,39),(30,48)]
 verts=[(x,y+s,z) for s in [-6,6] for x,z in profile];N=len(profile)
 faces=[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N)for i in range(N)]
 m=bpy.data.meshes.new('finger tab');m.from_pydata(verts,[],faces);m.update();t=bpy.data.objects.new('finger tab',m);bpy.context.collection.objects.link(t);clean(t)
 bpy.context.view_layer.objects.active=t;b=t.modifiers.new('soft tab edges','BEVEL');b.width=.8;b.segments=3;bpy.ops.object.modifier_apply(modifier=b.name)
 union(o,t)
rep=export(o,'magnet-carrier');(out/'geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
