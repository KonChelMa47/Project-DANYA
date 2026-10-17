exec(open('work/head-connectors-v3/build-shell.py').read().split('# Smooth radial')[0])
bpy.data.objects.remove(head,do_unlink=True)
bpy.ops.wm.stl_import(filepath=str(out/'head-neck-clean.stl'));head=bpy.context.object
bpy.ops.wm.stl_import(filepath=str(out/'mirror-integrated-addition.stl'));cradle=bpy.context.object;clean(cradle)
m=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];m[14]+=8;m=Matrix([[m[i+j*4]for j in range(4)]for i in range(4)])
for x in [-10,10]:
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=3.7,depth=4.2,location=(x,0,-23));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 for v in o.data.vertices:v.co=m@v.co
 boolean(cradle,o,'DIFFERENCE')
print('CRADLE CLEAR',export(cradle,'mirror-integrated-addition-cleared'),flush=True);boolean(head,cradle,'UNION')
report=json.load(open(out/'shell-report.json'))
exec(open('work/head-connectors-v3/build-shell.py').read().split('# Captive-nut mounting bosses')[1].split('\n',1)[1])
