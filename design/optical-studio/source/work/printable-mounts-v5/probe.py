exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(root/'outputs/danya-simulator/assets/head-connectors-v3/head-neck-clean.stl'));head=bpy.context.object;wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons])
d=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];d[14]+=8;m=Matrix([[d[i+4*j]for j in range(4)]for i in range(4)]);inv=m.inverted();dire=m.to_3x3()@Vector((0,0,-1))
rows=[]
for y in [-90,-84,-65,-45,0,45,65,84]:
 for x in [-40,-25,0,25,40]:
  h=wall.ray_cast(m@Vector((x,y,5)),dire,120)[0];rows.append({'xy':[x,y],'wallz':(inv@h).z if h else None,'world':list(m@Vector((x,y,-25)))})
(root/'work/printable-mounts-v5/head-probe.json').write_text(json.dumps(rows,indent=2));print(rows)
