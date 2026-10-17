import bpy,bmesh,json,shutil,numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
root=Path.cwd();r=root/'outputs/danya-simulator/assets/projection-prototype';v=root/'outputs/danya-simulator/assets/head-connectors-v3';prod=root/'outputs/danya-simulator/assets/buildable-bust-v10';backup=r/'archive-before-head-connectors-v3';backup.mkdir(exist_ok=True)
report=json.load(open(v/'shell-report.json'));assert report['head']['components']==1 and report['head']['nonManifoldEdges']==0
assert not json.load(open(v/'optical-check.json'))['newlyBlocked']
assert all(p['headOverlapMM3']<.01 for p in json.load(open(v/'mating-check.json')))
manifest=json.load(open(r/'parts.json'));validation=json.load(open(r/'print-validation.json'));c=json.load(open(v/'connectors.json'))
for n in ['parts.json','print-validation.json','ASSEMBLY.md','projection-prototype.blend','projection-prototype-print.zip','static-check.json','hardware-check.json','mating-volumes.json','final-print-audit.json','VERIFIED_FILES.json']:
 if not(backup/n).exists():shutil.copy2(r/n,backup/n)
for d in ['assembly','print']:
 (backup/d).mkdir(exist_ok=True)
 for n in ['head','head-arm-left','head-arm-right','mirror-head-cradle']:
  p=r/d/(n+'.stl')
  if p.exists()and not(backup/d/p.name).exists():shutil.copy2(p,backup/d/p.name)
pb=v/'archive-motorized';pb.mkdir(exist_ok=True)
for n in ['revised-structural-candidate.json','trial-left-head-rail.stl','trial-right-head-rail.stl']:
 if not(pb/n).exists():shutil.copy2(prod/n,pb/n)
manifest['parts']=[p for p in manifest['parts']if p['name']!='mirror-head-cradle'and not p['name'].startswith('head-connect-')];validation=[p for p in validation if p['name']not in ['head','head-arm-left','head-arm-right','mirror-head-cradle']]
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
if bpy.data.objects.get('mirror-head-cradle'):bpy.data.objects.remove(bpy.data.objects['mirror-head-cradle'],do_unlink=True)
for d in ['assembly','print']:
 p=r/d/'mirror-head-cradle.stl'
 if p.exists():p.unlink()
def load(p,name):
 bpy.ops.wm.stl_import(filepath=str(p));o=bpy.context.object;o.name=name;return o
def export(o,p):
 bpy.ops.object.select_all(action='DESELECT');o.hide_set(False);o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.wm.stl_export(filepath=str(p),export_selected_objects=True)
new=[]
for name,source in [('head','head-integrated'),('head-arm-left','static-head-arm-left'),('head-arm-right','static-head-arm-right')]:
 old=bpy.data.objects.get(name);mats=list(old.data.materials)if old else[]
 if old:bpy.data.objects.remove(old,do_unlink=True)
 shutil.copy2(v/(source+'.stl'),r/'assembly'/(name+'.stl'));o=load(r/'assembly'/(name+'.stl'),name)
 for mat in mats:o.data.materials.append(mat)
 new.append(o);p=next(p for p in manifest['parts']if p['name']==name);p['revision']='head-connectors-v3';p['notes']='Integrated mirror cradle and four captive M3 nut bosses; neck inner spikes trimmed, bottomZ50.65. Magnet holders retained.'if name=='head'else'Bolt-on head arm with two M3x8 + 3 washers per joint; M4 base interface retained. No head-support adhesive.'
for p in c['parts']:
 if not p['name'].startswith('head-connect-'):continue
 shutil.copy2(v/(p['name']+'.stl'),r/'assembly'/(p['name']+'.stl'));o=load(r/'assembly'/(p['name']+'.stl'),p['name']);manifest['parts'].append({**p,'print':False,'source':'head-connectors-v3 purchased hardware'})
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'))
for o in new:
 coords=np.array([x.co[:]for x in o.data.vertices]);lo=coords.min(0);hi=coords.max(0);M=Matrix.Translation(Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2])));dim=hi-lo
 for vert in o.data.vertices:vert.co=M@vert.co
 export(o,r/'print'/(o.name+'.stl'));bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bad=sum(not e.is_manifold for e in bm.edges);vol=bm.calc_volume(signed=False);bm.free();assert bad==0
 matrix=[M[i][j]for j in range(4)for i in range(4)];validation.append({'name':o.name,'dimensionsMM':dim.tolist(),'nonManifoldEdges':bad,'volumeMM3':vol,'fitsA1':bool(max(dim)<=256.001),'fitsAD5M':bool(max(dim)<=220.001),'assemblyToPrintMatrix':matrix});next(p for p in manifest['parts']if p['name']==o.name)['printTransform']=matrix
manifest['revision']='head-connectors-v3';(r/'parts.json').write_text(json.dumps(manifest,indent=2));(r/'print-validation.json').write_text(json.dumps(validation,indent=2))
# Production uses pre-lift coordinates; the viewer adds8mm to the roll group.
pman=json.load(open(prod/'revised-structural-candidate.json'));pman['parts']=[p for p in pman['parts']if p['name']!='mirror-head-cradle'and not p['name'].startswith(('head-mount-head-pad-','head-connect-'))]
for p in c['parts']:
 if p['name'].startswith('static-'):continue
 name=p['name'];dest='trial-'+name.split('-')[-1]+'-head-rail'if name.startswith('motorized-')else name;o=load(v/(name+'.stl'),'production-'+dest)
 for vert in o.data.vertices:vert.co.z-=8
 export(o,prod/(dest+'.stl'));entry={**p,'name':dest,'xy':[0,0],'revision':'head-connectors-v3'}
 if name.startswith('motorized-'):pman['parts']=[x for x in pman['parts']if x['name']!=dest]
 pman['parts'].append(entry)
 # Independent motorized replacement print files, no whole-system qualification implied.
 if name.startswith('motorized-'):
  (v/'motorized-print').mkdir(exist_ok=True);coords=np.array([x.co[:]for x in o.data.vertices]);lo=coords.min(0);hi=coords.max(0)
  for vert in o.data.vertices:vert.co-=Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]))
  export(o,v/'motorized-print'/(dest+'.stl'))
for name,path in [('head',v/'head-integrated.stl'),('face',r/'assembly/face.stl')]:
 o=load(path,'bin-'+name);bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.to_mesh(o.data);bm.free();vs=np.array([x.co[:]for x in o.data.vertices],dtype=np.float32);vs[:,2]-=8;fs=np.array([p.vertices[:]for p in o.data.polygons],dtype=np.uint32);vs.tofile(v/(name+'-prelift.positions.bin'));fs.tofile(v/(name+'-prelift.indices.bin'))
pman['headConnectorRevision']='head-connectors-v3';(prod/'revised-structural-candidate.json').write_text(json.dumps(pman,indent=2));print('ADOPTED both',len(validation),'staticprintfiles')
