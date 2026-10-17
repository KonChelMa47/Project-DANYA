import bpy,bmesh,json,shutil,numpy as np,math
from mathutils import Matrix,Vector
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'split-head-v6';backup=r/'archive-before-split-head-v6';backup.mkdir(exist_ok=True)
assert json.load(open(v/'assembly-check.json'))['newlyBlocked']==0
assert any(p['collisionCount']==0 for p in json.load(open(v/'mirror-access-check.json'))['routes'])
assert all(p['solutions']for p in json.load(open(v/'tool-access-check.json'))['access'])
assert not json.load(open(v/'tool-access-check.json'))['metalContacts']
assert all(not p['blockedSteps']for p in json.load(open(v/'nut-entry-check.json')))
assert json.load(open(v/'exterior-check.json'))['maxSurfaceDeviationMM']<.002
manifest=json.load(open(r/'parts.json'));validation=json.load(open(r/'print-validation.json'));oldMeta={p['name']:p.copy()for p in manifest['parts']};meta=json.load(open(v/'hardware.json'))['parts'];new=[{'name':'head','template':True,'print':False,'referenceOnly':True,'notes':'Assembly reference, includes filled pin interiors. Do not print; use head-upper/lower.'},{'name':'head-upper','print':True,'notes':'Upper rear-shell, integrated mirror cradle. Split face down for printing.'},{'name':'head-lower','print':True,'notes':'Lower rear-shell with arm connectors. Flip split face down for printing.'}]+[{**p,'print':p['name'].startswith('head-alignment-pin-')}for p in meta];names=[p['name']for p in new];removed=['head','mirror-removable-cradle']+[p['name']for p in manifest['parts']if p['name'].startswith('mirror-cartridge-')]
for n in ['parts.json','print-validation.json','ASSEMBLY.md','projection-prototype.blend','projection-prototype-print.zip','final-print-audit.json','VERIFIED_FILES.json','PRINT_LIST.csv','guide/assembly-guide.pdf']:
 p=backup/n;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():shutil.copy2(r/n,p)
for folder in ['assembly','print']:
 (backup/folder).mkdir(exist_ok=True)
 for name in set(names+removed):
  for ext in ['stl','step']:
   p=r/folder/(name+'.'+ext)
   if p.exists():
    if not(backup/folder/p.name).exists():shutil.copy2(p,backup/folder/p.name)
    if name in removed:p.unlink()
validation=[p for p in validation if p['name']not in names+removed];manifest['parts']=[p for p in manifest['parts']if p['name']not in names+removed]
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
for name in set(names+removed):
 if bpy.data.objects.get(name):bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
objects=[]
for entry in new:
 name=entry['name'];entry.update({'parent':'static','source':'split-head-v6','revision':'split-head-v6'});entry.setdefault('template',False)
 for ext in ['stl','step']:
  if(v/(name+'.'+ext)).exists():shutil.copy2(v/(name+'.'+ext),r/'assembly'/(name+'.'+ext))
 bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;o.name=name
 color=(.85,.89,.84,1)if name=='head-upper'else(.73,.81,.79,1)if name=='head-lower'else(.95,.6,.2,1)if'pin-'in name else(.5,.56,.6,1);mat=bpy.data.materials.new(name+'-v6');mat.diffuse_color=color;o.data.materials.append(mat);o.color=color
 if entry.get('referenceOnly',False):o.hide_render=True;o.hide_set(True)
 manifest['parts'].append(entry);objects.append((o,entry))
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
for o,p in objects:
 if not p['print']:continue
 M=Matrix.Rotation(math.pi,4,'X')if p['name']=='head-lower'else Matrix.Identity(4)
 for vertex in o.data.vertices:vertex.co=M@vertex.co
 coords=np.array([vv.co[:]for vv in o.data.vertices]);lo=coords.min(0);hi=coords.max(0);shift=Matrix.Translation(Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2])));M=shift@M
 for vertex in o.data.vertices:vertex.co=shift@vertex.co
 o.hide_set(False);bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(r/'print'/(o.name+'.stl')),export_selected_objects=True)
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bad=sum(not e.is_manifold for e in bm.edges);vol=bm.calc_volume(signed=False);bm.free();assert bad==0,(o.name,bad)
 transform=[M[i][j]for j in range(4)for i in range(4)];p['printTransform']=transform;dim=hi-lo;validation.append({'name':o.name,'dimensionsMM':dim.tolist(),'nonManifoldEdges':bad,'volumeMM3':vol,'fitsA1':bool(max(dim)<=256.001),'fitsAD5M':bool(max(dim)<=220.001),'assemblyToPrintMatrix':transform})
manifest['revision']='static-split-head-v6';(r/'parts.json').write_text(json.dumps(manifest,indent=2));(r/'print-validation.json').write_text(json.dumps(validation,indent=2));print('ADOPTED',len(validation),'print files')
