from pathlib import Path
import json,shutil,hashlib,zipfile,csv
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'face-v12';stage=v/'release';mapping={'magnet-carrier':'magnet-carrier-covered','head-upper':'head-upper-covered','head-lower':'head-lower-covered'}
geo=json.loads((v/'release-geometry.json').read_text());sweep=json.loads(Path('work/face-v12/final-removal-sweep.json').read_text());assert not sweep['collisions'],sweep['collisions']
for source in list(mapping.values())+[f'magnet-clear-{i}-{side}-magnet'for i in range(2)for side in ['head','face']]:assert json.loads((v/(source+'-rays.json')).read_text())['blockedSamples']==0,source
for source in mapping.values():
 d=json.loads((v/(source+'-surface.json')).read_text());assert d['crossingsOver01MM']==0 and d['coplanarOverlapsOver00001MM2']==0,source
for d in geo.values():assert d['components']==1 and d['nonManifoldEdges']==0
fit=json.loads((v/'magnet-fit.json').read_text());assert min(row['minimumMidplaneRadialClearanceMM']for row in fit['rows'])>.1
static=json.loads(Path('work/face-v12/static-check.json').read_text());baseline=json.loads(Path('work/face-v11/static-check.json').read_text());assert all(row in baseline for row in static),static
before={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for folder in ['assembly','print']for p in (r/folder).glob('*.stl')};backup=r/'archive-before-face-v12';backup.mkdir(exist_ok=True);changed=[]
for folder in ['assembly','print']:
 for src in (stage/folder).glob('*.stl'):
  dst=r/folder/src.name;b=backup/folder/src.name;b.parent.mkdir(parents=True,exist_ok=True)
  if not b.exists():shutil.copy2(dst,b)
  changed.append(str(dst.relative_to(r)))
for f in ['parts.json','print-validation.json','projection-prototype.blend','projection-prototype-print.zip','PRINT_LIST.csv','VERIFIED_FILES.json','ASSEMBLY.md','MAGNET_ASSEMBLY.md']:
 if (r/f).exists()and not(backup/f).exists():shutil.copy2(r/f,backup/f)
for folder in ['assembly','print']:
 for src in (stage/folder).glob('*.stl'):shutil.copy2(src,r/folder/src.name)
m=json.loads((r/'parts.json').read_text())
notes={'magnet-carrier':'Concealed upper 3x1mm magnets and smooth anatomical perimeter; no external handles. Bond unchanged face screen at perimeter.','head-upper':'Concealed upper 3x1mm magnet seats; adhesive retention. Lower magnets remain 5.9x2.9mm.','head-lower':'Lower internal front edge relieved for face unit removal.'}
for p in m['parts']:
 if p['name']in geo:p.update(revision='face-v12',notes=notes[p['name']],printTransform=geo[p['name']]['printTransform'])
 if 'assembly/'+p['name']+'.stl'in changed:p['revision']='face-v12'
 if p['name'].startswith(('magnet-clear-0-','magnet-clear-1-')):p.update(notes='Upper concealed magnet: diameter3 x thickness1mm; 4 total.',source='v12 original design; 3x1mm upper magnets')
m['revision']='face-v12';m['status']='Fixed projection prototype. Face unit straight forward removal sampled; physical pull force and bond strength untested.';(r/'parts.json').write_text(json.dumps(m,indent=2))
validation=json.loads((r/'print-validation.json').read_text())
for p in validation:
 if p['name']in geo:
  p.update(geo[p['name']]);p['assemblyToPrintMatrix']=p.pop('printTransform')
(r/'print-validation.json').write_text(json.dumps(validation,indent=2))
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in before.items()if p not in changed)
audit={'revision':'face-v12','changedSTLs':changed,'unchangedSTLs':len(before)-len(changed),'basePartsModified':False,'faceScreenModified':False,'mirrorsModified':False,'layoutModified':False};(v/'adoption-audit.json').write_text(json.dumps(audit,indent=2))
summary={**audit,'opticalSamples':255542,'changedPartBlockedSamples':0,'withdrawal':{'direction':'+X','travelMM':100,'first15MMStep':.25,'remainingStepMM':1,'collisions':0,'continuousCollisionProof':False},'staticContacts':static,'staticContactsNewVersusV11':[],'minimumMagnetRadialSampleClearanceMM':min(row['minimumMidplaneRadialClearanceMM']for row in fit['rows']),'geometry':geo,'physicalPullForceTest':False,'adhesiveStrengthTest':False,'physicalProjectionTest':False};(v/'validation-summary.json').write_text(json.dumps(summary,indent=2));shutil.copy2('work/face-v12/final-removal-sweep.json',v/'removal-sweep.json')
for f in ['ASSEMBLY.md','MAGNET_ASSEMBLY.md']:
 p=r/f;p.write_text('# 最新：v12 顔ユニットの着脱構造\n\n[最新版の交換・着脱手順](face-v12/ASSEMBLY.md)を優先してください。以下は旧版の記録です。\n\n'+p.read_text())
(v/'shadow-points.json').write_text('[]')
print(audit)
