from pathlib import Path
import json,zipfile,hashlib,shutil,csv,io
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'thick-head-v3';names=['head-upper','head-lower'];checks=['geometry-validation.json','components.json','wall-thickness-check.json','head-upper-surface.json','head-lower-surface.json','head-upper-rays.json','head-lower-rays.json','face-removal-sweep.json','magnet-fit.json','assembly-contact-check.json','pin-reuse.json','head-split-check.json','print-transforms.json','nut-entry-check.json']
for p in out.glob('*hardware*.json'):checks.append(p.name)
for n in ['head-inside','lower-inside','side']:shutil.copyfile(Path('work/thick-head')/(n+'.png'),out/(n+'.png'))
with zipfile.ZipFile(out/'DANYA-head-thick-center-v3.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for n in names:
  for ext in ['stl','3mf']:z.write(out/'print'/(n+'.'+ext),'MODELS/'+n+'.'+ext)
 z.write(out/'README-JA.md','README-JA.md')
 for n in checks:z.write(out/n,'VALIDATION/'+n)
 for n in ['head-inside.png','lower-inside.png','side.png']:z.write(out/n,'GUIDE/'+n)
old=r/'complete-latest/DANYA-prototype-ALL-PARTS-solid-seam-v2.zip';new=r/'complete-latest/DANYA-prototype-ALL-PARTS-thick-head-v3.zip'
with zipfile.ZipFile(old)as src,zipfile.ZipFile(new,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as dst:
 m=json.loads(src.read('MODEL_MANIFEST.json'))
 for info in src.infolist():
  p=info.filename;n=Path(p).stem
  if p.startswith('06_')or p=='MODEL_MANIFEST.json':continue
  if p=='README_FIRST_JA.md':dst.writestr(p,(out/'README-JA.md').read_text()+'\n全体セットは36印刷部品です。再印刷は上下の後頭部2個だけで、残る34部品のSTLは前版と同一です。旧ガイドの分割位置より本説明を優先してください。\n');continue
  if p=='PARTS.csv':
   rows=list(csv.reader(io.StringIO(src.read(p).decode('utf-8-sig'))))
   for row in rows[1:]:
    if Path(row[0]).stem in names:row[2]='thick-head-v3'
   text=io.StringIO();csv.writer(text).writerows(rows);dst.writestr(p,'\ufeff'+text.getvalue());continue
  if (p.startswith('01_PRINT/')or p.startswith('05_ORIENTED_3MF/'))and n in names:
   data=(out/'print'/Path(p).name).read_bytes();dst.writestr(p,data)
   if p.startswith('01_PRINT/'):
    for row in m['models']:
     if row['file']==p:row.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source=str(out/'print'/Path(p).name),revision='thick-head-v3')
  elif p in ['03_GUIDES/head-inside.png','03_GUIDES/lower-inside.png','03_GUIDES/side.png']:dst.write(out/Path(p).name,p)
  else:dst.writestr(p,src.read(p))
 m['seamRevision']='thick-head-v3';dst.writestr('MODEL_MANIFEST.json',json.dumps(m,ensure_ascii=False,indent=2));dst.write(out/'README-JA.md','03_GUIDES/THICK_HEAD_V3.md')
 for n in checks:dst.write(out/n,'06_GEOMETRY_VALIDATION/'+n)
with zipfile.ZipFile(new)as z,zipfile.ZipFile(old)as src:
 assert z.testzip()is None
 for row in m['models']:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['sha256']
 same=[p for p in src.namelist()if p.startswith('01_PRINT/')and Path(p).stem not in names];assert all(src.read(p)==z.read(p)for p in same)
 report={'changedPrintedParts':names,'unchangedPrintedParts':len(same),'verifiedModelHashes':len(m['models']),'zipBytes':new.stat().st_size,'slicerRun':False};(out/'package-verification.json').write_text(json.dumps(report,indent=2));print(report)
with zipfile.ZipFile(out/'DANYA-head-thick-center-v3.zip')as z:
 assert z.testzip()is None
 for n in names:assert z.read('MODELS/'+n+'.stl')==(out/(n+'.stl')).read_bytes()
