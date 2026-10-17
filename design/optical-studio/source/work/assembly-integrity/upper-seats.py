exec(open('work/assembly-integrity/seats.py').read().split("bpy.ops.wm.stl_import(filepath=str(out/'head-lower.stl'))")[0])
for name in ['head-upper','magnet-carrier']:
 bpy.ops.wm.stl_import(filepath=str(out/(name+'.stl')));obj=bpy.context.object
 for y in [-18.054,18.054]:
  if name=='head-upper':
   op(obj,cyl((40.15,y,255),2.9,2.9),'UNION');op(obj,box((40.0,y,257.8),(2.8,5,6.2)),'UNION');op(obj,cyl((41.5,y,255),1.65,2.3),'DIFFERENCE')
  else:
   op(obj,cyl((43.05,y,255),2.9,2.7),'UNION');op(obj,box((43.25,y,257.5),(2.3,5,5.8)),'UNION');op(obj,cyl((41.7,y,255),1.65,2.3),'DIFFERENCE')
 print(export(obj,name),flush=True);bpy.data.objects.remove(obj,do_unlink=True)
