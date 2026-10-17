exec(open('work/face-v9/fit-collar-screen.py').read().split("bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);face=load(out/'face-fused.stl');repair(face);reps={}
for n in ['head-upper','head-lower']:
 h=load(out/(n+'-finished.stl'));repair(h);fc=face.copy();fc.data=face.data.copy();bpy.context.collection.objects.link(fc);boolean(h,fc,'INTERSECT');repair(h);rep=export(h,n+'-face-overlap');reps[n]=rep;print(n,rep,flush=True)
(out/'face-head-overlaps.json').write_text(json.dumps(reps,indent=2))
