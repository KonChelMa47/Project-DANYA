from pathlib import Path
import json,csv,hashlib,zipfile,datetime
r=Path('outputs/danya-simulator/assets/projection-prototype');audit=json.load(open(r/'final-print-audit.json'));manifest=json.load(open(r/'parts.json'))
assert len(audit)==39 and all(x['connectedComponents']==1 and x['nonManifoldEdges']==0 and abs(x['bottomZ'])<.001 for x in audit)
assert all(hashlib.sha256((r/'print'/x['file']).read_bytes()).hexdigest()==x['sha256']for x in audit)
for f in ['static-check.json','hardware-check.json']:
 d=json.load(open(r/f));assert d['samples']==167698 and not d['newSupportRayObstructions']
assert all(p.get('overlapMM3',0)<1e-5 for p in json.load(open(r/'mating-volumes.json')))
assert not any(any('static-hardware' in n and ('bolt' in n or 'nut' in n) for n in pair)for pair in json.load(open(r/'static-check.json'))['surfaceContacts'])
with open(r/'PRINT_LIST.csv','w',encoding='utf-8-sig',newline='')as f:
 w=csv.writer(f);w.writerow(['STL','Quantity','Width mm','Depth mm','Height mm','A1','AD5M','Role'])
 for a in audit:w.writerow([a['file'],1,*[round(v,3)for v in a['dimensionsMM']],'fits','fits (check bed margin)'if max(a['dimensionsMM'])<=220.001 else'does not fit','cutting master'if 'master' in a['file']else'fit coupon'if 'coupon' in a['file']else'part'])
v=r/'split-head-v6'
assert json.load(open(v/'assembly-check.json'))['newlyBlocked']==0
assert any(p['collisionCount']==0 for p in json.load(open(v/'mirror-access-check.json'))['routes'])
assert all(not p['blockedSteps'] for p in json.load(open(v/'nut-entry-check.json')))
assert all(p['solutions'] for p in json.load(open(v/'tool-access-check.json'))['access'])
assert not json.load(open(v/'tool-access-check.json'))['metalContacts']
assert json.load(open(v/'exterior-check.json'))['maxSurfaceDeviationMM']<.002
v7=r/'prototype-v7'
latest=json.load(open(v7/'assembly-check.json'));assert latest['samples']==167698 and latest['missingPaths']==0 and not latest['changedPartRayObstructions']
assert not [p for p in latest['contacts']if not p['existing']]
assert any(p['collisions']==0 for p in json.load(open(v7/'mirror-access-check.json'))['routes'])
assert all(p['unsupportedDownwardAreaOver45degMM2']<.001 for p in json.load(open(v7/'print-check.json'))if p['name'].startswith('head-column'))
records={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for directory in ['print','assembly']for p in sorted((r/directory).glob('*.stl'))};(r/'VERIFIED_FILES.json').write_text(json.dumps({'timestampUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'physicalTests':'not performed','frameAndNewFastenerRaySamples':167698},indent=2))
files=list((r/'connector-checks').glob('*'))+list((r/'print').glob('*.stl'))+[r/n for n in ['ASSEMBLY.md','PRINT_LIST.csv','overview.png','open-head.png','mirror-cutting-outline.svg','foldMirror-cutting-outline.svg','print-validation.json','final-print-audit.json','static-check.json','hardware-check.json','mating-volumes.json','parts.json','layout.json','VERIFIED_FILES.json','MAGNET_ASSEMBLY.md','magnet-detail.png','magnets-v2/integration-report.json','magnets-v2/integration-check.json']]
files += [r/'guide/assembly-guide.pdf'] + [p for p in (r/'printable-mounts-v5').iterdir() if ('tray' in p.name.lower() or p.name=='TRAY_ASSEMBLY.md') and p.suffix in ['.json','.png','.md']] + [p for p in v.iterdir() if p.suffix in ['.json','.png','.md'] and p.name not in ['candidates.json','flange-optical-search.json','hardware.json']]
files += list((r/'mirror-guide').glob('*.png')) + [r/'mirror-guide/ASSEMBLY.md',r/'mirror-guide/mirror-assembly-guide.pdf']
files += [v7/n for n in ['ASSEMBLY.md','prototype-v7-guide.pdf','mirror-cutting-outline.svg','columns-print.png','column-hole-detail.png','parts.json','assembly-check.json','print-check.json','mirror-access-check.json']]
files += [r/'face-v8'/n for n in ['README.md','index.html','reinforced-face.png','shadow-map.svg','geometry-check.json','surface-check.json','optical-check.json']]
assert json.load(open(r/'face-v8/surface-check.json'))['face']['nonAdjacentCrossingsOver01MM']==0
with zipfile.ZipFile(r/'projection-prototype-print.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for p in files:z.write(p,'DANYA-static-prototype/'+str(p.relative_to(r)))
with zipfile.ZipFile(r/'projection-prototype-print.zip')as z:assert z.testzip()is None
print('ZIP',round((r/'projection-prototype-print.zip').stat().st_size/1e6,1),'MB',len(files),'files')

updateFiles=[r/'print'/f'{n}.stl' for n in ['head-upper','head-lower','head-alignment-pin-0','head-alignment-pin-1']]+[v/'ASSEMBLY.md',v/'split-overview.png',v/'joint-detail.png',v/'print-orientation.png',v/'head-split-check.json',v/'assembly-check.json',v/'nut-entry-check.json',r/'guide/assembly-guide.pdf',r/'final-print-audit.json']
with zipfile.ZipFile(v/'split-head-update-print.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in updateFiles:z.write(p,('print/' if p.suffix=='.stl' else '')+p.name)
with zipfile.ZipFile(v/'split-head-update-print.zip') as z:assert z.testzip() is None
print('HEAD UPDATE ZIP',len(updateFiles))

changedPrint=[p['name'] for p in json.load(open(v7/'print-check.json'))]
with zipfile.ZipFile(v7/'prototype-v7-update-print.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for n in changedPrint:z.write(r/'print'/(n+'.stl'),'print/'+n+'.stl')
 for n in ['ASSEMBLY.md','prototype-v7-guide.pdf','mirror-cutting-outline.svg','columns-print.png','column-hole-detail.png','assembly-check.json','print-check.json','mirror-access-check.json']:z.write(v7/n,n)
 z.write(r/'mirror-guide/mirror-assembly-guide.pdf','mirror-assembly-guide.pdf')
with zipfile.ZipFile(v7/'prototype-v7-update-print.zip')as z:assert z.testzip()is None
print('V7 UPDATE',len(changedPrint),'STLs')
