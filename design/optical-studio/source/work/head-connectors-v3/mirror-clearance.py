exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/head-connectors-v3';bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.wm.stl_import(filepath=str(out/'head-integrated.stl'));head=bpy.context.object
m=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];m[14]+=8;m=Matrix([[m[i+j*4]for j in range(4)]for i in range(4)])
for x in [-10,10]:
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=3.7,depth=3.9,location=(x,0,-23.05));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 for v in o.data.vertices:v.co=m@v.co
 boolean(head,o,'DIFFERENCE')
r=json.load(open(out/'shell-report.json'));r['head']=export(head,'head-integrated');r['mirrorBoltClearanceMM']=.2;(out/'shell-report.json').write_text(json.dumps(r,indent=2));print(r);assert r['head']['components']==1 and r['head']['nonManifoldEdges']==0
