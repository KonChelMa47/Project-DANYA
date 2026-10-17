exec(open('work/face-v10/build.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
d=np.load(r/'face-v8/screen-data.npz');v=d['after'].copy();f=d['outerFaces'].copy();apex=np.array(json.load(open('work/face-v9/virtual-source.json')));edges={}
for tri in f:
 for a,b in zip(tri,np.roll(tri,-1)):
  key=tuple(sorted((int(a),int(b))));edges.setdefault(key,[]).append((int(a),int(b)))
extra=[]
for es in edges.values():
 if len(es)==1:
  a,b=es[0];extra.append((b,a,len(v)))
# Stop 0.005 mm short of the original inner surface; preserve the skin and join.
ids=np.unique(f);delta=v[ids]-apex;v[ids]+=.01*delta/np.linalg.norm(delta,axis=1)[:,None]
m=bpy.data.meshes.new('inner optical cavity');m.from_pydata(np.vstack([v,apex]).tolist(),[],np.vstack([f,np.array(extra)]).tolist());m.update();c=bpy.data.objects.new('cavity',m);bpy.context.collection.objects.link(c);clean(c);print('cavity',export(c,'inner-cavity'),flush=True)
o=load(out/'face-joined-3.stl');bpy.context.view_layer.objects.active=o;mod=o.modifiers.new('clear light cavity','BOOLEAN');mod.operation='DIFFERENCE';mod.solver='MANIFOLD';mod.object=c;bpy.ops.object.modifier_apply(modifier=mod.name);bpy.data.objects.remove(c,do_unlink=True);print('carrier',export(o,'carrier-outer-clear'),flush=True)
skin=load(r/'face-v8/screen-after.stl');union(o,skin);print('united',export(o,'face-outer-carved'),flush=True)
