# Conformal adhesive feet connect the printable bridge to the existing head interior.
import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
root=Path.cwd();out=root/'outputs/danya-simulator/assets/buildable-bust-v10';base=root/'outputs/danya-simulator/assets/torso-flat-study'
v=np.fromfile(base/'head.positions.bin',np.float32).reshape(-1,3);f=np.fromfile(base/'head.indices.bin',np.uint32).reshape(-1,3);wall=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
d=json.load(open(root/'work/buildable-bust-v8/datums.json'))['mirror']['matrix'];m=Matrix([[d[i+4*j]for j in range(4)]for i in range(4)]);inv=m.inverted();direction=(m.to_3x3()@Vector((0,0,-1))).normalized()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(out/'mirror-head-bridge.stl'));bridge=bpy.context.object
reports=[]
for k,x in enumerate([-16,16]):
 # A 14x24mm pad each side, flared to the two corresponding bridge roots.
 verts=[];faces=[];count=9
 for layer in [0,1]:
  for i in range(count):
   for j in range(count):
    xx=x-7+14*i/(count-1);yy=-12+24*j/(count-1);start=m@Vector((xx,yy,-24));hit=wall.ray_cast(start,direction,100)[0]
    if hit is None:raise RuntimeError('no head interior at pad')
    h=inv@hit;zz=h.z+.2 if layer==0 else -30.5
    # Adhesive surface follows inner wall, with0.2mm bondline, root overlaps0.5mm.
    verts.append(tuple(m@Vector((xx,yy,zz))))
 N=count*count
 for layer in [0,1]:
  for i in range(count-1):
   for j in range(count-1):
    a=layer*N+i*count+j;faces.append((a,a+1,a+count+1,a+count))
 perimeter=[j for j in range(count)]+[i*count+count-1 for i in range(1,count)]+[(count-1)*count+j for j in range(count-2,-1,-1)]+[i*count for i in range(count-2,0,-1)]
 for a,b in zip(perimeter,perimeter[1:]+perimeter[:1]):faces.append((a,b,b+N,a+N))
 mesh=bpy.data.meshes.new('foot');mesh.from_pydata(verts,[],faces);mesh.update();bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(mesh);bm.free();foot=bpy.data.objects.new('foot',mesh);bpy.context.collection.objects.link(foot)
 bpy.context.view_layer.objects.active=bridge;mod=bridge.modifiers.new('Join conformal foot','BOOLEAN');mod.operation='UNION';mod.solver='EXACT';mod.object=foot;bpy.ops.object.modifier_apply(modifier=mod.name);bpy.data.objects.remove(foot,do_unlink=True)
 reports.append({'side':k,'widthMM':14,'heightMM':24,'bondlineMM':.2})
bpy.ops.object.select_all(action='DESELECT');bridge.select_set(True);bpy.context.view_layer.objects.active=bridge;bpy.ops.wm.stl_export(filepath=str(out/'mirror-head-cradle.stl'),export_selected_objects=True)
bm=bmesh.new();bm.from_mesh(bridge.data);report={'boundaryEdges':sum(e.is_boundary for e in bm.edges),'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volume':bm.calc_volume(signed=False)};bm.free();(out/'mirror-head-cradle.json').write_text(json.dumps({'status':'printed conformal bonded cradle candidate; adhesive load test pending','feet':reports,'mesh':report},indent=2));print(report)
