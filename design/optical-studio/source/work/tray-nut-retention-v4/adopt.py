import bpy,bmesh,json,shutil,numpy as np,math
from mathutils import Matrix,Vector
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'tray-nut-retention-v4';backup=r/'archive-before-tray-nut-retention-v4';backup.mkdir(exist_ok=True)
assert json.load(open(v/'assembly-check.json'))['newlyBlocked']==0
assert all(max(x[k]for k in ['trayStandOverlapMM3','coverStandOverlapMM3','maxNonDetentSlidingOverlapMM3'])<1e-5 for x in json.load(open(v/'mating-check.json')))
manifest=json.load(open(r/'parts.json'));validation=json.load(open(r/'print-validation.json'));meta=json.load(open(v/'parts.json'));pm=Matrix([[meta['matrix'][i+j*4]for j in range(4)]for i in range(4)]);inv=pm.inverted()
for n in ['parts.json','print-validation.json','ASSEMBLY.md','projection-prototype.blend','projection-prototype-print.zip','final-print-audit.json','VERIFIED_FILES.json','PRINT_LIST.csv','guide/assembly-guide.pdf']:
 p=backup/n;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():shutil.copy2(r/n,p)
for d in ['assembly','print']:
 (backup/d).mkdir(exist_ok=True)
 for name in ['drop-projector-tray-t5']+[f'drop-projector-tray-t5-frame-nut-{i}'for i in range(4)]:
  p=r/d/(name+'.stl')
  if p.exists()and not(backup/d/p.name).exists():shutil.copy2(p,backup/d/p.name)
replace=['drop-projector-tray-t5']+[f'drop-projector-tray-t5-frame-nut-{i}'for i in range(4)];newnames=[f'tray-nut-cover-{i}'for i in range(4)]+['tray-nut-fit-coupon'];names=replace+newnames
validation=[p for p in validation if p['name']not in names];oldMeta={p['name']:p for p in manifest['parts']};manifest['parts']=[p for p in manifest['parts']if p['name']not in names]
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'));objects=[]
for name in names:
 if bpy.data.objects.get(name):bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
 shutil.copy2(v/(name+'.stl'),r/'assembly'/(name+'.stl'));bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(name+'.stl')));o=bpy.context.object;o.name=name
 color=(.9,.57,.2,1)if name.startswith('tray-nut-cover')else(.24,.52,.54,1);mat=bpy.data.materials.new(name+'-v4');mat.diffuse_color=color;o.data.materials.append(mat)
 entry=next(p.copy()for p in meta['parts']if p['name']==name);entry.update({'print':not 'frame-nut-'in name and name!='tray-nut-fit-coupon','template':name=='tray-nut-fit-coupon','source':'static-only tray-nut-retention-v4','revision':'tray-nut-retention-v4'})
 if name=='tray-nut-fit-coupon':o.hide_render=True;o.hide_set(True);entry['fixture']=True
 manifest['parts'].append(entry);objects.append((o,entry))
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
for o,p in objects:
 if not p['print']and not p['template']:continue
 # Print in tray-localXZ plane, with undersideY=-8 facing the bed.
 M=Matrix.Rotation(math.pi/2,4,'X')@(Matrix.Identity(4)if p['template']else inv)
 for vertex in o.data.vertices:vertex.co=M@vertex.co
 coords=np.array([vv.co[:]for vv in o.data.vertices]);lo=coords.min(0);hi=coords.max(0);shift=Matrix.Translation(Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2])));M=shift@M
 for vertex in o.data.vertices:vertex.co=shift@vertex.co
 o.hide_set(False);bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(r/'print'/(o.name+'.stl')),export_selected_objects=True)
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bad=sum(not e.is_manifold for e in bm.edges);vol=bm.calc_volume(signed=False);bm.free();assert bad==0,(o.name,bad)
 transform=[M[i][j]for j in range(4)for i in range(4)];p['printTransform']=transform;dim=hi-lo;validation.append({'name':o.name,'dimensionsMM':dim.tolist(),'nonManifoldEdges':bad,'volumeMM3':vol,'fitsA1':bool(max(dim)<=256.001),'fitsAD5M':bool(max(dim)<=220.001),'assemblyToPrintMatrix':transform})
manifest['revision']='static-tray-nut-retention-v4';(r/'parts.json').write_text(json.dumps(manifest,indent=2));(r/'print-validation.json').write_text(json.dumps(validation,indent=2));print('ADOPTED',len(validation),'print files; motorized unchanged')
