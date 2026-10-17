exec(open('work/face-v12/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
o=load(out/'head-upper-covered.stl');clean(o)
for y in [-18.054,18.054]:
 stem=box((40.0,y,257.8),(3,4,3.6));intersect(stem,load(out/'head-envelope.stl'));union(o,stem)
 seat=cylinder_at((39.9,y,255),2.1,3);union(o,seat)
 cut(o,cylinder_at((41.05,y,255),1.65,1.5))
print(export(o,'head-upper-covered'),flush=True)
