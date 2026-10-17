import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
root=Path.cwd();out=root/'outputs/danya-simulator/assets/buildable-bust-v10';out.mkdir(exist_ok=True)
base=root/'outputs/danya-simulator/assets/torso-flat-study';fit=np.load(root/'work/seam_fit.npz')
def seam(y,z):
 y=y/1.1/100;z=(z/1.1-135)/100
 return 1.1*sum(c*y**i*z**j for c,(i,j) in zip(fit['coeff'],fit['powers']))
def mesh(name,v,f):
 m=bpy.data.meshes.new(name);m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o)
 bm=bmesh.new();bm.from_mesh(m);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(m);bm.free();return o
def apply(a,b,op):
 bpy.context.view_layer.objects.active=a;mod=a.modifiers.new(op,'BOOLEAN');mod.operation=op;mod.solver='EXACT';mod.object=b;bpy.ops.object.modifier_apply(modifier=mod.name)
def export(o,name):
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(out/(name+'.stl')),export_selected_objects=True)
 o.data.calc_loop_triangles();v=np.array([v.co[:] for v in o.data.vertices],np.float32);f=np.array([t.vertices[:] for t in o.data.loop_triangles],np.uint32);v.tofile(out/(name+'.positions.bin'));f.tofile(out/(name+'.indices.bin'))
 bm=bmesh.new();bm.from_mesh(o.data);report={'vertices':len(v),'triangles':len(f),'boundaryEdges':sum(e.is_boundary for e in bm.edges),'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'volume':bm.calc_volume(signed=False)};bm.free();return report
def cylinder(name,center,axis,r,length):
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=r,depth=length,location=center);o=bpy.context.object;o.name=name;o.rotation_mode='QUATERNION';o.rotation_quaternion=Vector(axis).to_track_quat('Z','Y');bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
 # Bake location for consistent raw exports.
 o.data.transform(o.matrix_world);o.matrix_world.identity();return o
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
shells={};vs=[];fs=[]
for name in ['head','face']:
 v=np.fromfile(base/(name+'.positions.bin'),np.float32).reshape(-1,3);f=np.fromfile(base/(name+'.indices.bin'),np.uint32).reshape(-1,3)
 shells[name]=mesh(name,v.tolist(),f.tolist());fs.extend((f+len(vs)).tolist());vs.extend(v.tolist())
wall=BVHTree.FromPolygons(vs,fs,all_triangles=True);datums=json.load(open(root/'work/buildable-bust-v10/magnet-datums.json'));rows=[]
for row in datums:
 sign=1 if row['seam'][1]>0 else -1;y=row['seam'][1]-sign*5.5;z=row['seam'][2];x=seam(y,z);center=Vector((x,y,z));eps=.01;n=Vector((1,-(seam(y+eps,z)-seam(y-eps,z))/(2*eps),-(seam(y,z+eps)-seam(y,z-eps))/(2*eps))).normalized()
 # Conformal foot follows the first inner surface hit, leaves0.15mm adhesive gap inside wall.
 count=13;verts=[];faces=[]
 for layer in [0,1]:
  for i in range(count):
   xx=x-6+12*i/(count-1)
   for j in range(count):
    zz=z-6+12*j/(count-1);hit=wall.ray_cast(Vector((xx,0,zz)),Vector((0,sign,0)),100)[0]
    if hit is None:raise RuntimeError('Missing wall '+row['id'])
    yy=hit.y-sign*.15 if layer else row['seam'][1]-sign*10
    verts.append((xx,yy,zz))
 N=count*count
 for layer in [0,1]:
  for i in range(count-1):
   for j in range(count-1):
    a=layer*N+i*count+j;faces.append((a,a+1,a+count+1,a+count))
 perimeter=[j for j in range(count)]+[i*count+count-1 for i in range(1,count)]+[(count-1)*count+j for j in range(count-2,-1,-1)]+[i*count for i in range(count-2,0,-1)]
 for a,b in zip(perimeter,perimeter[1:]+perimeter[:1]):faces.append((a,b,b+N,a+N))
 common=mesh('boss-'+row['id'],verts,faces)
 pair=[]
 for key,front in [('head',False),('face',True)]:
  ob=common.copy();ob.data=common.data.copy();bpy.context.collection.objects.link(ob);ob.name='magnet-seat-'+row['id']+'-'+key
  for v in ob.data.vertices:v.co.x-=seam(v.co.y,v.co.z)
  bm=bmesh.new();bm.from_mesh(ob.data);res=bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),dist=1e-6,plane_co=(0,0,0),plane_no=(1,0,0),clear_inner=front,clear_outer=not front)
  edges=[e for e in bm.edges if e.is_boundary];bmesh.ops.holes_fill(bm,edges=edges,sides=0);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(ob.data);bm.free()
  for v in ob.data.vertices:v.co.x+=seam(v.co.y,v.co.z)
  bore=cylinder('bore',center,n,3.075,6.6);apply(ob,bore,'DIFFERENCE');bpy.data.objects.remove(bore,do_unlink=True)
  report=export(ob,ob.name);ob.color=(.12,.55,.45,1)
  direction=1 if front else -1;mag=cylinder('magnet-'+row['id']+'-'+key,center+n*(direction*1.60),n,2.95,2.9);mr=export(mag,mag.name);mag.color=(.6,.65,.7,1);pair.append({'shell':key,'seat':ob.name,'magnet':mag.name,'seatMesh':report,'magnetMesh':mr})
 bpy.data.objects.remove(common,do_unlink=True)
 rows.append({'id':row['id'],'center':list(center),'axis':list(n),'magnetDiameterMM':5.9,'magnetThicknessMM':2.9,'pocketDiameterMM':6.15,'pocketDepthMM':3.3,'magnetRecessMM':.15,'parts':pair});print('BUILT',row['id'],flush=True)
report={k:export(o,k+'-magnetic') for k,o in shells.items()}
(out/'magnet-mounts.json').write_text(json.dumps({'status':'Separate conformal cartridges bonded inside original shells; retention/adhesive, optical and collision qualification pending','pairs':rows,'shellMeshes':report,'seam':'Original 1.1-scaled polynomial surface retained. No offset between shells; magnets recessed0.15mm nominal each side.'},indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(out/'magnetic-shells.blend'));print(json.dumps(report),flush=True)
