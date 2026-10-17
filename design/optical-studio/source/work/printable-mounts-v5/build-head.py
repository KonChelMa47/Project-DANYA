exec(open('work/projection-prototype/magnets-v2/build.py').read().split("\nbpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)")[0])
out=root/'outputs/danya-simulator/assets/projection-prototype/printable-mounts-v5';old=root/'outputs/danya-simulator/assets/head-connectors-v3'
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(old/'head-neck-clean.stl'));head=bpy.context.object;clean(head)
wall=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons]);report={}
# Preserve the four established head/arm connectors without integrating the mirror cradle.
source=open('work/head-connectors-v3/build-shell.py').read().split('# Captive-nut mounting bosses')[1].split('\n',1)[1].split("report['head']=")[0]
source=source.replace("out/'connectors.json'","old/'connectors.json'")
exec(source)
d=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];d[14]+=8;mm=Matrix([[d[i+4*j]for j in range(4)]for i in range(4)]);mi=mm.inverted();direction=mm.to_3x3()@Vector((0,0,-1))
def cube_local(name,lo,hi):
 bpy.ops.mesh.primitive_cube_add(size=1,location=[(lo[i]+hi[i])/2 for i in range(3)]);o=bpy.context.object;o.name=name;o.dimensions=[hi[i]-lo[i]for i in range(3)];bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 for v in o.data.vertices:v.co=mm@v.co
 return o
def cyl_local(x,y,z,h,r,vertices=64):
 bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=h,location=(x,y,z+h/2),rotation=(0,0,math.pi/6 if vertices==6 else 0));o=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 for v in o.data.vertices:v.co=mm@v.co
 return o
bpy.ops.wm.stl_import(filepath=str(old/'mirror-integrated-addition-cleared.stl'));cradle=bpy.context.object;cradle.name='mirror-removable-cradle';clean(cradle)
boolean(cradle,cube_local('rear-trim',[-200,-200,-200],[200,200,-38]),'DIFFERENCE')
# Four shell pads follow the inside wall and meet a common removable-cartridge plane.
anchors=[]
for x in [-16,16]:
 for y in [-7,7]:
  vs=[];n=9;zs=[]
  for layer in [0,1]:
   for i in range(n):
    for j in range(n):
     xx=x-5.5+11*i/(n-1);yy=y-5.5+11*j/(n-1);hit=wall.ray_cast(mm@Vector((xx,yy,-20)),direction,100)[0];assert hit is not None;z=(mi@hit).z;zs.append(z);vs.append(mm@Vector((xx,yy,z-.35 if layer==0 else-38)))
  boolean(head,grid(vs,n),'UNION')
  # Flat counterbore seat is reached from the outside of the rear shell.
  boolean(head,cyl_local(x,y,-100,56,3.75),'DIFFERENCE');boolean(head,cyl_local(x,y,-44.1,6.5,1.7),'DIFFERENCE')
  boolean(cradle,cyl_local(x,y,-32.6,19,5.8/math.sqrt(3),6),'DIFFERENCE');boolean(cradle,cyl_local(x,y,-38.1,5.6,1.7),'DIFFERENCE')
  anchors.append({'xy':[x,y],'shellWallLocalZ': [min(zs),max(zs)],'interfaceZ':-38,'counterboreSeatZ':-44,'screw':'M3x16','underHeadZ':-44.5,'screwTipZ':-28.5,'nutPocketFloorZ':-32.6,'nutZ':[-32.5,-30.1]})
report={'head':export(head,'head'),'cradle':export(cradle,'mirror-removable-cradle'),'anchors':anchors,'mirrorMatrix':d,'physicalTests':False,'scope':'Static prototype; shell exterior unchanged except four recessed access holes. Mirror optical pose and arm/magnet interfaces retained.'}
(out/'head-check.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
assert all(report[n]['components']==1 and report[n]['nonManifoldEdges']==0 for n in ['head','cradle'])
