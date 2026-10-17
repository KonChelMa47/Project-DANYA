exec(open('work/face-v9/fit-collar-screen.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);collar=load(out/'collar-exact-main.stl');repair(collar)
bm=bmesh.new();bm.from_mesh(collar.data);assert all(e.is_manifold for e in bm.edges);bm.free()
# 0.01mm local fusion overlap removes coincident Boolean surfaces at the face join.
# Pocket datums and the rear division plane remain fixed.
a=np.empty(len(collar.data.vertices)*3);collar.data.vertices.foreach_get('co',a);a=a.reshape(-1,3);fit=np.load('work/seam_fit.npz');y=a[:,1]/110;z=((a[:,2]-8)/1.1-135)/100;sx=1.1*sum(c*y**i*z**j for c,(i,j)in zip(fit['coeff'],fit['powers']));d=.01*np.clip((a[:,0]-sx+2)/1.2,0,1);a[:,0]+=d;collar.data.vertices.foreach_set('co',a.ravel());collar.data.update();print('local overlap',float(d.max()),int((d>0).sum()),flush=True)
face=load(r/'face-v8/screen-after.stl');boolean(face,collar,'UNION');repair(face);rep=export(face,'face-fused');(out/'fused-geometry.json').write_text(json.dumps(rep,indent=2));print(rep,flush=True)
