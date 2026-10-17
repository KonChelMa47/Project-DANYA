exec(open('work/face-v11/redesign.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
o=load(out/'magnet-carrier-revised.stl');h=load(out/'head-upper-fixed.stl');lo=load(out/'head-lower-release.stl')
for y in [-18.054,18.054]:
 cut(o,box((38.2,y,260),(20,17,20))) # rearward lips above z252 removed, crown at seam retained
 cut(h,box((53.4,y,261),(10,20,20))) # front opening x48.4+
for y in [-29,29]:cut(lo,box((29,y,53.5),(7,5,8)))
for name,obj in [('magnet-carrier-revised',o),('head-upper-release',h),('head-lower-release',lo)]:print(name,export(obj,name),flush=True)
