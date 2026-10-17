import bpy,json,math
from pathlib import Path
from mathutils import Vector,Matrix
from bpy_extras.object_utils import world_to_camera_view
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';out=r/'mirror-guide';dat=json.load(open('work/buildable-bust-v8/datums.json'));report={}
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='OBJECT';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.show_shadows=True;s.display.shading.background_type='WORLD';s.world.color=(.96,.97,.98);s.view_settings.view_transform='Standard';s.render.resolution_x=1300;s.render.resolution_y=1100;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.clip_end=3000;s.camera=cam
colors=[(.32,.39,.47,1),(.65,.73,.81,1),(.12,.48,.52,1),(.69,.36,.7,1),(.24,.62,.65,1),(.91,.5,.19,1),(.89,.76,.4,1)]
def render(name,objs,cp,target,scale,anchors={}):
 for o in bpy.data.objects:
  if o.type=='MESH':o.hide_render=o not in objs
 cam.location=cp;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale;bpy.context.view_layer.update();s.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
 report[name]={k:list(world_to_camera_view(s,cam,Vector(v)))[:2]for k,v in anchors.items()}
for key,tag,prefix,lift in [('mirror','large','printed-mirror-',8),('foldMirror','small','drop-printed-foldMirror-',2)]:
 m=dat[key]['matrix'].copy();m[14]+=lift;M=Matrix([[m[j*4+i]for j in range(4)]for i in range(4)]);inv=M.inverted();items={};groups={};base={}
 names=[p.stem for p in (r/'assembly').glob('*.stl')if p.stem.startswith(prefix)]+[key+'-glass']
 for name in names:
  bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;o.name=tag+'-'+name
  for v in o.data.vertices:v.co=inv@v.co
  g=0 if 'adjust-bolt' in name else 1 if 'adjust-washer' in name else 2 if 'fixed-plate' in name or 'frame-nut' in name else 3 if 'elastic' in name else 4 if 'backing' in name or 'adjust-nut' in name else 5 if 'bond-pad' in name else 6
  o.color=colors[g];items[name]=o;groups[name]=g;base[name]=o.location.copy()
 # Separate along the actual mirror normal, preserving XY screw alignment.
 gap=30 if tag=='large'else 12;offsets=[-gap*2,-gap*1.25,0,gap,gap,gap*2,gap*3]
 for n,o in items.items():o.location.z=offsets[groups[n]]
 maxx=max(abs(v.co.x)for o in items.values()for v in o.data.vertices);anchors={}
 for g in range(7):
  vs=[o.matrix_world@v.co for n,o in items.items()if groups[n]==g for v in o.data.vertices];anchors[str(g+1)]=[maxx+8,0,(min(v.z for v in vs)+max(v.z for v in vs))/2] if vs else[0,0,0]
 # Update matrix_world before projecting labels.
 bpy.context.view_layer.update()
 anchors={str(i+1):[maxx+5,0,sum((vv.co.z+o.location.z)for n,o in items.items()if groups[n]==g for vv in o.data.vertices)/sum(len(o.data.vertices)for n,o in items.items()if groups[n]==g)]for i,g in enumerate([0,1,2,4,5,6])}
 render(tag+'-exploded',list(items.values()),(320,-500,240),(0,0,gap),360 if tag=='large'else 170,anchors)
 # Pocket entry close-ups, only plate plus its nuts shifted out of their entry faces.
 for n,o in items.items():o.location.z=0
 fixed=[o for n,o in items.items()if groups[n]==2]
 for n,o in items.items():
  if 'frame-nut' in n:o.location.z=10 if tag=='large'else 7;o.color=(.95,.64,.22,1)
 render(tag+'-fixed-nuts',fixed,(140,-200,230),(0,0,-10),110 if tag=='large'else 50)
 backing=[o for n,o in items.items()if groups[n]==4]
 for n,o in items.items():
  if 'adjust-nut' in n:o.location.z=-14 if tag=='large'else-7;o.color=(.95,.64,.22,1)
 render(tag+'-backing-nuts',backing,(140,-200,-230),(0,0,-8),200 if tag=='large'else 65)
 # Actual mounted view: frame bolts and shells/stand in assembled world coordinates.
 for n,o in items.items():
  o.location=(0,0,0)
  for vv in o.data.vertices:vv.co=M@vv.co
 extra=['head-upper','mirror-head-bolt-0','mirror-head-bolt-1','mirror-head-washer-0','mirror-head-washer-1']if tag=='large'else['lower-mirror-stand']
 if tag=='small':
  extra += [p['name']for p in json.load(open(r/'parts.json'))['parts']if p['name'].startswith('static-hardware-fold-')]
 for name in extra:
  if not(r/'assembly'/(name+'.stl')).exists():continue
  bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;o.color=(.7,.77,.78,1);items[name]=o
 render(tag+'-mounted',list(items.values()),(330,-360,240)if tag=='large'else(230,-300,120),(-25,0,220)if tag=='large'else(-35,0,-95),250 if tag=='large'else 150)
 # Mounting detail: preserve CAD coordinates, show support crop, frame,2 bolts and nuts only.
 import bmesh
 detail=[]
 support='head-upper'if tag=='large'else'lower-mirror-stand'
 for name,o in items.items():
  use=('fixed-plate'in name or 'frame-nut'in name or name==support or name.startswith('mirror-head-') or name.startswith('static-hardware-fold-'))
  if not use:continue
  for vv in o.data.vertices:vv.co=inv@vv.co
  if name==support:
   bm=bmesh.new();bm.from_mesh(o.data)
   # Illustration-only cutaway, the original STL remains unchanged.
   for co,no in [((35,0,0),(-1,0,0)),((-35,0,0),(1,0,0)),((0,25,0),(0,-1,0)),((0,-25,0),(0,1,0)),((0,0,-38),(0,0,1))]:
    bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),plane_co=co,plane_no=no,clear_inner=True,clear_outer=False)
   # Remove detached slivers created only by the illustration crop.
   unseen=set(bm.verts)
   while unseen:
    comp=[];stack=[unseen.pop()]
    while stack:
     vv=stack.pop();comp.append(vv)
     for e in vv.link_edges:
      nn=e.other_vert(vv)
      if nn in unseen:unseen.remove(nn);stack.append(nn)
    if len(comp)<150:bmesh.ops.delete(bm,geom=comp,context='VERTS')
   bm.to_mesh(o.data);bm.free();o.color=(.67,.72,.74,1)
  elif 'fixed-plate'in name or 'frame-nut'in name:o.location.z=22 if tag=='large'else 14
  else:o.location.z=-32
  detail.append(o)
 render(tag+'-mount-detail',detail,(150,-200,150),(0,0,-10),160 if tag=='large'else 110)
(out/'render-labels.json').write_text(json.dumps(report,indent=2))
