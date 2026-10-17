import bpy,bmesh,json,shutil,numpy as np,math
from mathutils import Matrix,Vector
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'printable-mounts-v5';backup=r/'archive-before-printable-mounts-v5';backup.mkdir(exist_ok=True)
assert json.load(open(v/'assembly-check.json'))['newlyBlocked']==0
checks=json.load(open(v/'head-mating-check.json'));assert checks['headCradleOverlapMM3']<.01
assert any(x['normalOffsetMM']==20 and x['collisionCount']==0 for x in json.load(open(v/'mirror-removal-search.json'))['candidates'])
manifest=json.load(open(r/'parts.json'));validation=json.load(open(r/'print-validation.json'));meta=json.load(open(v/'tray-parts.json'));pm=Matrix([[meta['matrix'][i+j*4]for j in range(4)]for i in range(4)]);inv=pm.inverted();md=json.load(open(v/'head-check.json'))['mirrorMatrix'];mm=Matrix([[md[i+j*4]for j in range(4)]for i in range(4)])
for n in ['parts.json','print-validation.json','ASSEMBLY.md','projection-prototype.blend','projection-prototype-print.zip','final-print-audit.json','VERIFIED_FILES.json','PRINT_LIST.csv','guide/assembly-guide.pdf']:
 p=backup/n;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():shutil.copy2(r/n,p)
new=[p.copy()for p in meta['parts']if p['name']!='tray-nut-cover-local']+json.load(open(v/'head-hardware.json'))['parts'];oldMeta={p['name']:p.copy()for p in manifest['parts']};new += [{**oldMeta['head'],'name':'head'},{'name':'mirror-removable-cradle','parent':'static','material':'printed PETG','notes':'Removable mirror cartridge. Four exterior M3x16 screws; front-loaded M3 nuts. Assemble mirror before inserting.'}];names=[p['name']for p in new]
for folder in ['assembly','print']:
 (backup/folder).mkdir(exist_ok=True)
 for name in names:
  for ext in ['stl','step']:
   p=r/folder/(name+'.'+ext)
   if p.exists()and not(backup/folder/p.name).exists():shutil.copy2(p,backup/folder/p.name)
validation=[p for p in validation if p['name']not in names];manifest['parts']=[p for p in manifest['parts']if p['name']not in names]
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'));objects=[]
for entry in new:
 name=entry['name']
 if bpy.data.objects.get(name):bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
 for ext in ['stl','step']:
  if(v/(name+'.'+ext)).exists():shutil.copy2(v/(name+'.'+ext),r/'assembly'/(name+'.'+ext))
 bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;o.name=name
 entry.update({'print':name in ['head','mirror-removable-cradle','drop-projector-tray-t5']or name.startswith('tray-nut-cover-'),'template':name=='tray-nut-fit-coupon','source':'static-only printable-mounts-v5','revision':'printable-mounts-v5'})
 color=(.95,.6,.2,1)if name.startswith('tray-nut-cover')or name=='mirror-removable-cradle'else(.79,.83,.81,1)if name=='head'else(.24,.52,.54,1)if entry['print']else(.6,.65,.67,1)
 mat=bpy.data.materials.new(name+'-v5');mat.diffuse_color=color;o.data.materials.append(mat);o.color=color
 if entry['template']:o.hide_render=True;o.hide_set(True);entry['fixture']=True
 manifest['parts'].append(entry);objects.append((o,entry))
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
for o,p in objects:
 if not p['print']and not p['template']:continue
 if p['name']=='head':
  d=oldMeta['head']['printTransform'];M=Matrix([[d[i+j*4]for j in range(4)]for i in range(4)])
 elif p['name']=='mirror-removable-cradle':M=mm.inverted()
 else:M=Matrix.Rotation(math.pi/2,4,'X')@(Matrix.Identity(4)if p['template']else inv)
 for vert in o.data.vertices:vert.co=M@vert.co
 coords=np.array([vv.co[:]for vv in o.data.vertices]);lo=coords.min(0);hi=coords.max(0);shift=Matrix.Translation(Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2])));M=shift@M
 for vert in o.data.vertices:vert.co=shift@vert.co
 o.hide_set(False);bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(r/'print'/(o.name+'.stl')),export_selected_objects=True)
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bad=sum(not e.is_manifold for e in bm.edges);vol=bm.calc_volume(signed=False);bm.free();assert bad==0,(o.name,bad)
 transform=[M[i][j]for j in range(4)for i in range(4)];p['printTransform']=transform;dim=hi-lo;validation.append({'name':o.name,'dimensionsMM':dim.tolist(),'nonManifoldEdges':bad,'volumeMM3':vol,'fitsA1':bool(max(dim)<=256.001),'fitsAD5M':bool(max(dim)<=220.001),'assemblyToPrintMatrix':transform})
manifest['revision']='static-printable-mounts-v5';(r/'parts.json').write_text(json.dumps(manifest,indent=2));(r/'print-validation.json').write_text(json.dumps(validation,indent=2));print('ADOPTED',len(validation),'print files')
