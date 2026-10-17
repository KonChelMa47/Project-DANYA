from pathlib import Path
import json,shutil,hashlib,zipfile
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'face-v10';stage=v/'release';geo=json.loads((v/'release-geometry.json').read_text())
for n in ['magnet-carrier-pockets','head-upper-final','head-lower-final']:
 d=json.loads((v/(n+'-surface.json')).read_text());assert d['crossingsOver01MM']==0 and d['coplanarOverlapsOver00001MM2']==0
 d=json.loads((v/(n+'-rays.json')).read_text());assert d['blockedSamples']==0
for d in geo.values():assert d['components']==1 and d['nonManifoldEdges']==0
fit=json.loads((v/'magnet-fit.json').read_text());assert min(x['minimumMidplaneRadialClearanceMM'] for x in fit['rows'])>.05
contacts=json.loads((v/'assembly-check.json').read_text());assert not contacts['carrierOtherPartContacts'] and not contacts['headContacts']
backup=r/'archive-before-face-v10';backup.mkdir(exist_ok=True)
changed=[]
for folder in ['assembly','print']:
 for src in (stage/folder).glob('*.stl'):
  dst=r/folder/src.name
  if dst.exists():
   b=backup/folder/src.name;b.parent.mkdir(parents=True,exist_ok=True)
   if not b.exists():shutil.copy2(dst,b)
  changed.append(str(dst.relative_to(r)))
for f in ['parts.json','print-validation.json','projection-prototype.blend','projection-prototype-print.zip','ASSEMBLY.md','MAGNET_ASSEMBLY.md']:
 if (r/f).exists() and not (backup/f).exists():shutil.copy2(r/f,backup/f)
for folder in ['assembly','print']:
 for src in (stage/folder).glob('*.stl'):shutil.copy2(src,r/folder/src.name)
m=json.loads((r/'parts.json').read_text());notes={'face':'Original v8 thickened screen, old magnet arms removed. Bond to separate magnet-carrier at perimeter. Exterior and screen thickness retained.','magnet-carrier':'Separate peripheral magnetic carrier, top/bottom four 5.9x2.9mm magnets, epoxy retention; bond to face perimeter. Nominal sampled ray clearance verified; physical bond strength untested.','head-upper':'Revised upper magnetic receivers; mirror mounting unchanged.','head-lower':'Revised lower magnetic receivers; stand mounting unchanged.'}
for name,d in geo.items():
 p=next((p for p in m['parts'] if p['name']==name),None)
 if p is None:p={'name':name,'print':True,'material':'printed PETG','template':False};m['parts'].append(p)
 p.update({'revision':'face-v10','notes':notes[name],'printTransform':d['printTransform']})
for p in m['parts']:
 if 'assembly/'+p['name']+'.stl' in changed:p['revision']='face-v10'
m['revision']='face-v10';m['status']='Static prototype revision. Sampled optical/mesh checks passed; adhesive joint and physical tests pending.';(r/'parts.json').write_text(json.dumps(m,indent=2))
validation=json.loads((r/'print-validation.json').read_text())
for name,d in geo.items():
 p=next((p for p in validation if p['name']==name),None)
 if p is None:p={'name':name};validation.append(p)
 p.update(d);p['assemblyToPrintMatrix']=p.pop('printTransform')
(r/'print-validation.json').write_text(json.dumps(validation,indent=2))
before=json.loads(Path('work/head-light-clearance/before-sha256.json').read_text());unexpected=[p for p,h in before.items() if p not in changed and hashlib.sha256((r/p).read_bytes()).hexdigest()!=h];assert not unexpected
report={'changedSTLs':changed,'unchangedOriginalSTLs':sum(p not in changed for p in before),'unexpectedChangedSTLs':unexpected,'basePartsModified':False,'faceScreenSource':'face-v8/screen-after.stl','layoutModified':False,'physicalStrengthTest':False};(v/'adoption-audit.json').write_text(json.dumps(report,indent=2));print(report)
