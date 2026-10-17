import bpy,bmesh,json,shutil,numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'magnets-v2';backup=r/'archive-before-integrated-magnets';backup.mkdir(exist_ok=True)
report=json.load(open(v/'integration-report.json'));check=json.load(open(v/'integration-check.json'))
assert all(x['components']==1 and x['nonManifoldEdges']==0 for x in report['shells'].values())
assert not check['opposingPartSurfaceContacts'] and check['maximumAddedVertexOutsideOriginalShellEnvelopeMM']<.001
manifest=json.load(open(r/'parts.json'));validation=json.load(open(r/'print-validation.json'))
for name in ['parts.json','print-validation.json','projection-prototype-print.zip','ASSEMBLY.md']:
 if not(backup/name).exists():shutil.copy2(r/name,backup/name)
for d in ['assembly','print']:
 (backup/d).mkdir(exist_ok=True)
 for name in ['head','face']+[f'magnet-clear-{i}-{s}'for i in range(4)for s in ['face-return','head-cup']]:
  p=r/d/(name+'.stl')
  if p.exists()and not(backup/d/p.name).exists():shutil.copy2(p,backup/d/p.name)
remove={f'magnet-clear-{i}-{s}'for i in range(4)for s in ['face-return','head-cup']}
manifest['parts']=[p for p in manifest['parts']if p['name']not in remove and p['name']!='magnet-fit-coupon'];validation=[p for p in validation if p['name']not in remove and p['name']not in ['head','face','magnet-fit-coupon']]
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
for name in remove:
 if name in bpy.data.objects:bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
 for d in ['assembly','print']:
  p=r/d/(name+'.stl')
  if p.exists():p.unlink()
def load(p,name):
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;o.name=name;return o
def export(o,p):
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(p),export_selected_objects=True)
new=[]
for name in ['head','face']:
 old=bpy.data.objects[name];materials=list(old.data.materials);bpy.data.objects.remove(old,do_unlink=True);shutil.copy2(v/(name+'-integrated.stl'),r/'assembly'/(name+'.stl'));o=load(r/'assembly'/(name+'.stl'),name)
 for m in materials:o.data.materials.append(m)
 colName='顔：一体型磁石ホルダー'if name=='face'else'後頭部：一体型磁石ホルダー';col=bpy.data.collections.get(colName)or bpy.data.collections.new(colName)
 if not col.name in bpy.context.scene.collection.children:bpy.context.scene.collection.children.link(col)
 for obj in [o]+[bpy.data.objects.get(f'magnet-clear-{i}-{name}-magnet')for i in range(4)]:
  if obj:
   for c in list(obj.users_collection):c.objects.unlink(obj)
   col.objects.link(obj)
 new.append(o);p=next(p for p in manifest['parts']if p['name']==name);p['notes']='Four integrated magnet holders; Ø6.15 bore,Ø6.75 epoxy key,3 axial lands. Original exterior retained. Epoxy retention, physical pull test pending.';p['revision']='integrated-magnets-v2'
shutil.copy2(v/'magnet-fit-coupon.stl',r/'assembly/magnet-fit-coupon.stl');coupon=load(v/'magnet-fit-coupon.stl','magnet-fit-coupon');coupon.hide_render=True;coupon.hide_set(True);new.append(coupon);manifest['parts'].append({'name':'magnet-fit-coupon','print':False,'template':True,'fixture':True,'material':'PETG','notes':'1/2/3/4 dimples correspond toØ6.00/6.15/6.30/6.45mm. Dry fit before full shell print.'})
# Keep printable master and assembled parts at different coordinate systems.
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
for o in new:
 coords=np.array([x.co[:]for x in o.data.vertices]);lo=coords.min(0);hi=coords.max(0);M=Matrix.Translation(Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2])));dim=hi-lo
 for vert in o.data.vertices:vert.co=M@vert.co
 o.hide_set(False);export(o,r/'print'/(o.name+'.stl'))
 bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bad=sum(not e.is_manifold for e in bm.edges);volume=bm.calc_volume(signed=False);bm.free();assert bad==0
 matrix=[M[i][j]for j in range(4)for i in range(4)];validation.append({'name':o.name,'dimensionsMM':dim.tolist(),'nonManifoldEdges':bad,'volumeMM3':volume,'fitsA1':bool(max(dim)<=256.001),'fitsAD5M':bool(max(dim)<=220.001),'assemblyToPrintMatrix':matrix});next(p for p in manifest['parts']if p['name']==o.name)['printTransform']=matrix
manifest['revision']='integrated-magnets-v2';(r/'parts.json').write_text(json.dumps(manifest,indent=2));(r/'print-validation.json').write_text(json.dumps(validation,indent=2));print('ADOPTED',len(validation),'printSTLs')
