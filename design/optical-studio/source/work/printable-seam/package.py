from pathlib import Path
import zipfile,hashlib,json
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'printable-seam';names=['head-upper','head-lower','magnet-carrier'];validation=['FINAL-VALIDATION.json','actual-wall-slice-verification.json','actual-head-transects.json','face-removal-sweep.json','magnet-fit.json']+[n+'-rays.json'for n in names]
with zipfile.ZipFile(out/'DANYA-printable-seam-update.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 for n in names:
  for ext in ['stl','3mf']:z.write(out/'print'/(n+'.'+ext),'MODELS/'+n+'.'+ext)
 z.write(out/'README-JA.md','README-JA.md')
 for n in ['assembled.png','exploded.png','seam-detail.png','verified-slice-paths.png']:z.write(out/n,'GUIDE/'+n)
 for n in validation:z.write(out/n,'VALIDATION/'+n)
old=r/'complete-latest/DANYA-prototype-ALL-PARTS-2026-09-25.zip';new=r/'complete-latest/DANYA-prototype-ALL-PARTS-seam-v1.zip';changed=[]
with zipfile.ZipFile(old)as src,zipfile.ZipFile(new,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as dst:
 manifest=json.loads(src.read('MODEL_MANIFEST.json'));README=src.read('README_FIRST_JA.md').decode();README='''# 合わせ目補強版 seam-v1 を採用しています

後頭部上下と顔の支持枠は新しい補強版へ交換済みです。最初に03_GUIDES/PRINTABLE_SEAM.mdを読んでください。後頭部上下と支持枠のSTLは組立座標です。05_ORIENTED_3MFの各モデルには印刷方向とベッド配置を設定しています。材料・プリンター設定は別途適用してください。以下の旧セット説明で「顔と磁石支持枠はv12」「後頭部は分割面を下にしたSTL」とする箇所より本説明を優先します。

'''+README
 for info in src.infolist():
  path=info.filename
  if path=='MODEL_MANIFEST.json':continue
  if path=='README_FIRST_JA.md':dst.writestr(path,README);continue
  if path=='PARTS.csv':
   import csv,io
   rows=list(csv.reader(io.StringIO(src.read(path).decode('utf-8-sig'))))
   for row in rows[1:]:
    if Path(row[0]).stem in names:row[2]='printable-seam-v1'
   txt=io.StringIO();csv.writer(txt).writerows(rows);dst.writestr(path,'\ufeff'+txt.getvalue());continue
  stem=Path(path).stem
  if path.startswith('01_PRINT/')and stem in names:
   data=(out/'print'/(stem+'.stl')).read_bytes();dst.writestr(path,data);changed.append(path)
   for row in manifest['models']:
    if row['file']==path:row.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source=str(out/'print'/(stem+'.stl')),revision='printable-seam-v1')
  else:dst.writestr(path,src.read(path))
 manifest['seamRevision']='printable-seam-v1';dst.writestr('MODEL_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));dst.write(out/'README-JA.md','03_GUIDES/PRINTABLE_SEAM.md')
 for n in names:dst.write(out/'print'/(n+'.3mf'),'05_ORIENTED_3MF/'+n+'.3mf')
 for n in validation:dst.write(out/n,'06_SEAM_VALIDATION/'+n)
 for n in ['assembled.png','exploded.png','seam-detail.png','verified-slice-paths.png']:dst.write(out/n,'03_GUIDES/seam-'+n)
with zipfile.ZipFile(new)as z,zipfile.ZipFile(old)as o:
 assert z.testzip()is None
 for row in manifest['models']:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['sha256']
 same=[p for p in o.namelist()if p.startswith('01_PRINT/')and p not in changed];assert all(o.read(p)==z.read(p)for p in same)
 report={'zip':str(new),'changedPrintedParts':changed,'unchangedPrintedParts':len(same),'verifiedModelHashes':len(manifest['models']),'bytes':new.stat().st_size};(out/'package-verification.json').write_text(json.dumps(report,indent=2));print(report)
