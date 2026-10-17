import bpy,bmesh,numpy as np,json,math
from pathlib import Path
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
root=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';manifest=json.load(open(root/'parts.json'))
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def load(n):
 bpy.ops.wm.stl_import(filepath=str(root/'assembly'/(n+'.stl')));o=bpy.context.object;o.name=n;bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(o.data);bm.free();return o
def export(o,path):
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(path),export_selected_objects=True)
def union(o,b):
 bpy.context.view_layer.objects.active=o;mod=o.modifiers.new('Conformal feet','BOOLEAN');mod.operation='UNION';mod.solver='EXACT';mod.object=b;bpy.ops.object.modifier_apply(modifier=mod.name);bpy.data.objects.remove(b,do_unlink=True)
head=load('head');tree=BVHTree.FromPolygons([v.co for v in head.data.vertices],[p.vertices[:]for p in head.data.polygons])
for sg in [-1,1]:
 o=load('head-arm-'+('left'if sg<0 else'right'))
 for pad in json.load(open(Path.cwd()/'work/buildable-bust-v8/head-pads.json')):
  if pad['point'][1]*sg<0:continue
  p=Vector(pad['point']);p.z+=8;n=Vector(pad['normal']);u=Vector((-n.y,n.x,0));verts=[];faces=[];N=9
  for layer in [0,1]:
   for i in range(N):
    for j in range(N):
     start=p+u*(-5+10*i/(N-1))+Vector((0,0,-5+10*j/(N-1)))-12*n
     hit=tree.ray_cast(start,n,30)[0]
     if hit is None:raise RuntimeError('Missing head shoe hit')
     verts.append(tuple(hit-.25*n if layer else start+4*n))
  for layer in [0,1]:
   for i in range(N-1):
    for j in range(N-1):
     a=layer*N*N+i*N+j;faces.append((a,a+1,a+N+1,a+N))
  perimeter=list(range(N))+[i*N+N-1 for i in range(1,N)]+[(N-1)*N+j for j in range(N-2,-1,-1)]+[i*N for i in range(N-2,0,-1)]
  for a,b in zip(perimeter,perimeter[1:]+perimeter[:1]):faces.append((a,b,b+N*N,a+N*N))
  me=bpy.data.meshes.new('foot');me.from_pydata(verts,[],faces);me.update();bm=bmesh.new();bm.from_mesh(me);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(me);bm.free();obj=bpy.data.objects.new('foot',me);bpy.context.collection.objects.link(obj);union(o,obj)
 export(o,root/'assembly'/(o.name+'.stl'))
# Import remaining objects. Store the true assembly, then export individually oriented print STLs.
objects={o.name:o for o in bpy.context.scene.objects}
for p in manifest['parts']:
 if p['name']not in objects:objects[p['name']]=load(p['name'])
colors={'PETG':(.12,.42,.47,1),'head':(.8,.82,.79,1),'face':(.9,.85,.73,1),'projector':(.5,.12,.1,1)}
for p in manifest['parts']:
 o=objects[p['name']];c=colors.get(o.name,colors['PETG'] if p.get('print')else(.6,.57,.45,1));mat=bpy.data.materials.new(o.name);mat.diffuse_color=c;o.data.materials.append(mat);o.hide_render=p.get('template',False);o.hide_set(p.get('template',False))
bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=.001
bpy.ops.wm.save_as_mainfile(filepath=str(root/'projection-prototype.blend'))
report=[];dat=json.load(open('work/buildable-bust-v8/datums.json'));pm=json.load(open(root/'generated-parts.json'))['projectorBaseMatrix']
def mat(a):return Matrix([[a[i+4*j]for j in range(4)]for i in range(4)])
for p in manifest['parts']:
 if not(p.get('print')or p.get('template')):continue
 o=objects[p['name']];bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.to_mesh(o.data)
 bad=sum(not e.is_manifold for e in bm.edges);volume=bm.calc_volume(signed=False);bm.free()
 coords=np.array([v.co[:]for v in o.data.vertices]);rots=[Matrix.Identity(4),Matrix.Rotation(math.pi/2,4,'X'),Matrix.Rotation(-math.pi/2,4,'X'),Matrix.Rotation(math.pi/2,4,'Y'),Matrix.Rotation(-math.pi/2,4,'Y'),Matrix.Rotation(math.pi,4,'X')]
 if o.name.startswith('drop-projector-tray'):rots=[Matrix.Rotation(math.pi/2,4,'X')@mat(pm).inverted()]
 elif o.name.startswith('printed-mirror'):m=dat['mirror']['matrix'].copy();m[14]+=8;rots=[mat(m).inverted()]
 elif o.name.startswith('drop-printed-foldMirror'):m=dat['foldMirror']['matrix'].copy();m[14]+=2;rots=[mat(m).inverted()]
 elif o.name in ['head','face']:rots=[Matrix.Identity(4)]
 options=[]
 for r in rots:
  v=np.array([tuple(r@Vector(c))for c in coords]);dim=np.ptp(v,axis=0);options.append((dim[2],r,v,dim))
 _,r,v,dim=min(options,key=lambda t:t[0]);lo=v.min(0);hi=v.max(0);shift=Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]));M=Matrix.Translation(shift)@r
 for vert in o.data.vertices:vert.co=M@vert.co
 o.hide_set(False);export(o,root/'print'/(o.name+'.stl'))
 row={'name':o.name,'dimensionsMM':dim.tolist(),'nonManifoldEdges':bad,'volumeMM3':volume,'fitsA1':bool(max(dim)<=256.001),'fitsAD5M':bool(max(dim)<=220.001),'assemblyToPrintMatrix':[M[i][j]for j in range(4)for i in range(4)]};report.append(row)
 p['printTransform']=row['assemblyToPrintMatrix']
(root/'print-validation.json').write_text(json.dumps(report,indent=2));(root/'parts.json').write_text(json.dumps(manifest,indent=2));print('PRINT_REPORT',json.dumps(report))
