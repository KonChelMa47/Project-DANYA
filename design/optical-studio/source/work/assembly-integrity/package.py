from pathlib import Path
import json,zipfile,hashlib,shutil,csv,io
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'solid-seam-v2';names=['head-upper','head-lower'];checks=['geometry-validation.json','head-upper-surface.json','head-lower-surface.json','head-upper-rays.json','head-lower-rays.json','face-removal-sweep.json','magnet-fit.json','neck-smoothing.json']
for src,dst in [('components-v2.json','all-parts-components.json'),('zip-geometry.json','previous-zip-all-parts-geometry.json'),('part-proximity.json','all-parts-proximity-before.json')]:shutil.copyfile(Path('work/assembly-integrity')/src,out/dst);checks.append(dst)
for src,dst in [('head-inside.png','before-head-inside.png'),('lower-inside.png','before-lower-inside.png'),('v2-head-inside.png','head-inside.png'),('v2-lower-inside.png','lower-inside.png'),('v2-side.png','side.png')]:shutil.copyfile(Path('work/assembly-integrity')/src,out/dst)
with zipfile.ZipFile(out/'DANYA-head-repair-v2.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for n in names:
  for ext in ['stl','3mf']:z.write(out/'print'/(n+'.'+ext),'MODELS/'+n+'.'+ext)
 z.write(out/'README-JA.md','README-JA.md')
 for n in checks:z.write(out/n,'VALIDATION/'+n)
 for n in ['head-inside.png','lower-inside.png','side.png']:z.write(out/n,'GUIDE/'+n)
old=r/'complete-latest/DANYA-prototype-ALL-PARTS-seam-v1.zip';new=r/'complete-latest/DANYA-prototype-ALL-PARTS-solid-seam-v2.zip'
with zipfile.ZipFile(old)as src,zipfile.ZipFile(new,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as dst:
 m=json.loads(src.read('MODEL_MANIFEST.json'))
 for info in src.infolist():
  p=info.filename;n=Path(p).stem
  if p.startswith('06_SEAM_VALIDATION/')or p.startswith('03_GUIDES/seam-'):continue
  if p=='MODEL_MANIFEST.json':continue
  if p=='README_FIRST_JA.md':dst.writestr(p,(out/'README-JA.md').read_text()+'\n\n全体セット:01_PRINTの36部品を各1個、02_TEMPLATES_AND_COUPONSの4個は組立部品ではなく確認用です。旧ガイドの形状よりこの修正版を優先してください。\n');continue
  if p=='PARTS.csv':
   rows=list(csv.reader(io.StringIO(src.read(p).decode('utf-8-sig'))))
   for row in rows[1:]:
    if Path(row[0]).stem in names:row[2]='solid-seam-v2'
   text=io.StringIO();csv.writer(text).writerows(rows);dst.writestr(p,'\ufeff'+text.getvalue());continue
  if (p.startswith('01_PRINT/')or p.startswith('05_ORIENTED_3MF/'))and n in names:
   data=(out/'print'/Path(p).name).read_bytes();dst.writestr(p,data)
   if p.startswith('01_PRINT/'):
    for row in m['models']:
     if row['file']==p:row.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source=str(out/'print'/Path(p).name),revision='solid-seam-v2')
  elif p=='03_GUIDES/PRINTABLE_SEAM.md':dst.write(out/'README-JA.md',p)
  else:dst.writestr(p,src.read(p))
 m['seamRevision']='solid-seam-v2';dst.writestr('MODEL_MANIFEST.json',json.dumps(m,ensure_ascii=False,indent=2))
 for n in checks:dst.write(out/n,'06_GEOMETRY_VALIDATION/'+n)
 for n in ['head-inside.png','lower-inside.png','side.png']:dst.write(out/n,'03_GUIDES/'+n)
with zipfile.ZipFile(new)as z,zipfile.ZipFile(old)as src:
 assert z.testzip()is None
 for row in m['models']:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['sha256']
 same=[p for p in src.namelist()if p.startswith('01_PRINT/')and Path(p).stem not in names];assert all(src.read(p)==z.read(p)for p in same)
 report={'changedPrintedParts':names,'unchangedPrintedParts':len(same),'verifiedModelHashes':len(m['models']),'zipBytes':new.stat().st_size,'slicerRun':False};(out/'package-verification.json').write_text(json.dumps(report,indent=2));print(report)
