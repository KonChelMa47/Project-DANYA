import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
root=Path('outputs/danya-simulator/assets/buildable-bust-v10')
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def mesh(name,v,f):
 m=bpy.data.meshes.new(name);m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(m);bm.free();return o
def boolean(a,b,operation):
 bpy.context.view_layer.objects.active=a;m=a.modifiers.new(operation,'BOOLEAN');m.operation=operation;m.solver='EXACT';m.object=b;bpy.ops.object.modifier_apply(modifier=m.name);bpy.data.objects.remove(b,do_unlink=True)
def cylinder(name,x,y,z,r,h,vertices=96):
 bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=h,location=(x,y,z+h/2));o=bpy.context.object;o.name=name;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);return o
v=np.fromfile(root/'body-lowered-floor.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(root/'body-lowered-floor.indices.bin',np.uint32).reshape(-1,3);wall=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);body=mesh('body-with-floor-bosses',v.tolist(),f.tolist());rows=[]
for index,(x,y,dx,dy) in enumerate([(-75,0,-1,0),(20,-85,0,-1),(-10,-98,0,-1),(-10,98,0,1),(20,85,0,1)]):
 vv=[];ff=[]
 for i in range(5):
  z=-156.7+7*i/4;ends=[]
  for side in [-4,4]:
   p=Vector((x-side*dy,y+side*dx,z));h=wall.ray_cast(p,Vector((dx,dy,0)),100)[0];assert h is not None;ends.append((p,h+Vector((dx,dy,0))*1.0))
  vv +=[tuple(ends[0][0]),tuple(ends[0][1]),tuple(ends[1][1]),tuple(ends[1][0])]
 for i in range(4):
  for j in range(4):ff.append((4*i+j,4*i+(j+1)%4,4*(i+1)+(j+1)%4,4*(i+1)+j))
 ff +=[(3,2,1,0),(16,17,18,19)];boolean(body,mesh('contoured-boss-web',vv,ff),'UNION');boolean(body,cylinder('boss',x,y,-156.7,6.5,7),'UNION');boolean(body,cylinder('M4-clearance',x,y,-156.8,2.2,7.2),'DIFFERENCE')
 # Explicit AF7.2 hex orientation matches the CAD nut envelope.
 rr=7.2/math.sqrt(3);vhex=[(x+rr*math.cos(j*math.pi/3),y+rr*math.sin(j*math.pi/3),z)for z in[-153.05,-149.5]for j in range(6)];fhex=[tuple(reversed(range(6))),tuple(range(6,12))]+[(j,(j+1)%6,(j+1)%6+6,j+6)for j in range(6)];boolean(body,mesh('M4-captive-pocket',vhex,fhex),'DIFFERENCE');rows.append({'center':[x,y],'bossBottomZ':-156.7,'bossTopZ':-149.7,'radiusMM':6.5,'pocketAFMM':7.2,'webWidthMM':8})
 bm=bmesh.new();bm.from_mesh(body.data);print('BOSS',index,'nonmanifold',sum(not e.is_manifold for e in bm.edges),flush=True);bm.free()
bm=bmesh.new();bm.from_mesh(body.data);seen=set();components=0
for p in bm.verts:
 if p in seen:continue
 components+=1;seen.add(p);q=[p]
 while q:
  a=q.pop()
  for e in a.link_edges:
   b=e.other_vert(a)
   if b not in seen:seen.add(b);q.append(b)
report={'components':components,'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volumeMM3':bm.calc_volume(signed=False),'bosses':rows,'scope':'Topology only; self-intersection, actual fit, tool clearance and strength qualification pending. Original upper-body variant retained, bottom extended4.5mm.'};bm.free();assert components==1 and report['nonManifoldEdges']==0,report
body.data.calc_loop_triangles();np.array([v.co[:]for v in body.data.vertices],np.float32).tofile(root/'body-with-floor-bosses.positions.bin');np.array([t.vertices[:]for t in body.data.loop_triangles],np.uint32).tofile(root/'body-with-floor-bosses.indices.bin');bpy.ops.object.select_all(action='DESELECT');body.select_set(True);bpy.context.view_layer.objects.active=body;bpy.ops.wm.stl_export(filepath=str(root/'body-with-floor-bosses.stl'),export_selected_objects=True);(root/'body-with-floor-bosses.json').write_text(json.dumps(report,indent=2));print(report)
