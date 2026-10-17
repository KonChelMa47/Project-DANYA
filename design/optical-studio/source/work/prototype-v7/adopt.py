import bpy,json,shutil
from pathlib import Path
r=Path.cwd()/'outputs/danya-simulator/assets/projection-prototype';v=r/'prototype-v7';backup=r/'archive-before-prototype-v7';backup.mkdir(exist_ok=True)
a=json.load(open(v/'assembly-check.json'));assert a['missingPaths']==0 and not a['changedPartRayObstructions'];assert not[p for p in a['contacts']if not p['existing']];assert any(p['collisions']==0 for p in json.load(open(v/'mirror-access-check.json'))['routes'])
checks=json.load(open(v/'print-check.json'));assert all(p['components']==1 and p['nonManifoldEdges']==0 for p in checks)
meta=json.load(open(v/'parts.json'));manifest=json.load(open(r/'parts.json'));old={p['name']:p for p in manifest['parts']};deleted=[n for n in old if '-elastic-'in n];newnames=[p['name']for p in meta['parts']]
for n in ['parts.json','layout.json','generated-parts.json','print-validation.json','ASSEMBLY.md','projection-prototype.blend','projection-prototype-print.zip','mirror-cutting-outline.svg','guide/assembly-guide.pdf','mirror-guide/mirror-assembly-guide.pdf','mirror-guide/ASSEMBLY.md','final-print-audit.json','PRINT_LIST.csv','VERIFIED_FILES.json']:
 dest=backup/n;dest.parent.mkdir(parents=True,exist_ok=True)
 if(r/n).exists()and not dest.exists():shutil.copy2(r/n,dest)
for folder in ['assembly','print']:
 (backup/folder).mkdir(exist_ok=True)
 for n in newnames+deleted:
  p=r/folder/(n+'.stl')
  if p.exists():
   if not(backup/folder/p.name).exists():shutil.copy2(p,backup/folder/p.name)
   if n in deleted:p.unlink()
for p in meta['parts']:
 n=p['name'];entry={**old[n],**p,'source':'prototype-v7','revision':'prototype-v7'}
 if n in [x['name']for x in checks]:entry['printTransform']=next(x['assemblyToPrintMatrix']for x in checks if x['name']==n)
 old[n]=entry
 for ext in ['stl','step']:
  if(v/(n+'.'+ext)).exists():shutil.copy2(v/(n+'.'+ext),r/'assembly'/(n+'.'+ext))
 if(v/'print'/(n+'.stl')).exists():shutil.copy2(v/'print'/(n+'.stl'),r/'print'/(n+'.stl'))
manifest['parts']=[old[p['name']]for p in manifest['parts']if p['name']not in deleted];manifest['revision']='static-prototype-v7';(r/'parts.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(v/'layout.json',r/'layout.json')
val=json.load(open(r/'print-validation.json'))
for row in checks:
 oldrow=next(p for p in val if p['name']==row['name']);oldrow.update(row);oldrow['fitsA1']=max(row['dimensionsMM'])<=256;oldrow['fitsAD5M']=max(row['dimensionsMM'])<=220
(r/'print-validation.json').write_text(json.dumps(val,indent=2))
g=json.load(open(r/'generated-parts.json'))
for t in g['mirrorTemplates']:
 t['glassThicknessAssumedMM']=1.5;t['glassThicknessUserMM']=1.5
 if t['name']=='mirror':t.update({'pointsMM':meta['mirrorPolygonMM'],'widthMM':meta['mirrorBoundingSizeMM'][0],'heightMM':meta['mirrorBoundingSizeMM'][1]})
for p in g['parts']:
 if p['name']in old and p['name']in newnames:p.update(old[p['name']])
(r/'generated-parts.json').write_text(json.dumps(g,indent=2))
bpy.ops.wm.open_mainfile(filepath=str(r/'projection-prototype.blend'))
for n in newnames+deleted:
 if bpy.data.objects.get(n):bpy.data.objects.remove(bpy.data.objects[n],do_unlink=True)
for p in meta['parts']:
 n=p['name'];bpy.ops.wm.stl_import(filepath=str(r/'assembly'/(n+'.stl')));o=bpy.context.object;o.name=n;color=(.3,.57,.59,1)if old[n].get('print')else(.9,.76,.4,1)if'glass'in n else(.6,.64,.67,1);o.color=color;mat=bpy.data.materials.new(n+'-v7');mat.diffuse_color=color;o.data.materials.append(mat)
 if old[n].get('template'):o.hide_render=True;o.hide_set(True)
bpy.ops.wm.save_as_mainfile(filepath=str(r/'projection-prototype.blend'));print('ADOPTED',len(newnames),'changed',len(deleted),'sleeves removed')
