from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
r=Path('outputs/danya-simulator/assets/projection-prototype');o=r.parent/'small-mirror-study';archive=r/'archive-before-small-mirror-v1';archive.mkdir(exist_ok=True)
v=json.loads((o/'candidate-validation.json').read_text());assert all(x['blocked']==0 for x in v['checks']);assert v['face']['missingGeometricPaths']==0
contacts=json.loads((o/'contacts.json').read_text())[-1] # contacts-outline script updates contacts.json
assert contacts['name']=='beveled-outline';assert all('foldMirror-bond-pad-'in n for n in contacts['hits'])
assert json.loads((o/'extra-mirror-path-check.json').read_text())['blocked']==0
report={}
for n in ['foldMirror-glass','foldMirror-cutting-master']:
 raw=np.fromfile(o/(n+'.stl'),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')],offset=84)['v'];verts,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3);e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);_,c=np.unique(e,axis=0,return_counts=True);assert np.all(c==2),(n,c[c!=2]);vol=np.einsum('ij,ij->i',verts[f[:,0]],np.cross(verts[f[:,1]],verts[f[:,2]])).sum()/6;assert vol>0;report[n]={'triangles':len(f),'boundaryOrNonManifoldEdges':int((c!=2).sum()),'signedVolumeMM3':float(vol),'boundsMM':[verts.min(0).tolist(),verts.max(0).tolist()]}
files=['layout.json','parts.json','assembly/foldMirror-glass.stl','assembly/foldMirror-cutting-master.stl','print/foldMirror-cutting-master.stl','foldMirror-cutting-outline.svg','projection-prototype-print.zip','projection-prototype.blend']
for f in files:
 if(r/f).exists()and not(archive/f).exists():(archive/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/f,archive/f)
before={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for folder in ['assembly','print']for p in(r/folder).glob('*.stl')}
shutil.copy2(o/'candidate-layout.json',r/'layout.json');shutil.copy2(o/'foldMirror-glass.stl',r/'assembly/foldMirror-glass.stl')
for folder in ['assembly','print']:shutil.copy2(o/'foldMirror-cutting-master.stl',r/folder/'foldMirror-cutting-master.stl')
shutil.copy2(o/'foldMirror-cutting-outline.svg',r/'foldMirror-cutting-outline.svg')
d=json.loads((r/'parts.json').read_text())
for p in d['parts']:
 if p['name']=='foldMirror-glass':p.update(notes='Small-mirror-v1: 49x100mm envelope, one straight corner cut. Thickness1.5mm. Same optical plane and mount datum. No real pupil/ghost/strength certification.',revision='small-mirror-v1',source='small-mirror-study')
 if p['name']=='foldMirror-cutting-master':p['notes']='Small-mirror-v1 49x100mm pentagon. Print100%. Two small holes mark existing mount datum and +Y; do NOT drill glass. Glass1.5mm. See SMALL_MIRROR_UPDATE.md.'
(r/'parts.json').write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
changed=[p for p,h in before.items()if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h];assert set(changed)=={'assembly/foldMirror-glass.stl','assembly/foldMirror-cutting-master.stl','print/foldMirror-cutting-master.stl'}
report['changedSTL']=changed;report['unchangedSTLCount']=len(before)-len(changed);report['scope']='Only small mirror and its cutting master changed; all printed mechanism STLs including small mirror leg unchanged.';(o/'adoption-audit.json').write_text(json.dumps(report,indent=2));print(report)
