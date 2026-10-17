from pathlib import Path
import json,hashlib,zipfile,csv,datetime
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'face-v12';geo=json.loads((v/'release-geometry.json').read_text())
p=r/'PRINT_LIST.csv'
with p.open(encoding='utf-8-sig',newline='')as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
for row in rows:
 name=Path(row['STL']).stem
 if name in geo:
  d=geo[name];row.update({'Width mm':round(d['dimensionsMM'][0],3),'Depth mm':round(d['dimensionsMM'][1],3),'Height mm':round(d['dimensionsMM'][2],3),'A1':'fits','AD5M':'fits (check bed margin)','Role':'v12 detachable face assembly'})
with p.open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
checks={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for folder in ['assembly','print']for p in (r/folder).glob('*.stl')};(r/'VERIFIED_FILES.json').write_text(json.dumps({'timestampUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':'face-v12','scope':'Hashes only, not physical certification. See face-v12/validation-summary.json.','files':checks},indent=2))
sweep=json.loads((v/'removal-sweep.json').read_text());sweep.pop('stepMM',None);sweep['stepRanges']=[{'fromMM':0,'toMM':15,'stepMM':.25},{'fromMM':15,'toMM':100,'stepMM':1}];(v/'removal-sweep.json').write_text(json.dumps(sweep,indent=2))
readme='最新版はface-v12/ASSEMBLY.mdです。顔ユニットを工具なしで前へ外す構造へ更新。旧ガイドPDF・旧版検査は参考資料です。印刷前にv12手順を優先してください。磁力・接着強度・実物投影は未試験です。\n'
(r/'README_FIRST.md').write_text(readme)
with zipfile.ZipFile(v/'face-v12-update-print.zip','w',zipfile.ZIP_DEFLATED)as z:
 for n in geo:z.write(r/'print'/(n+'.stl'),'print/'+n+'.stl')
 for name in ['covered.png','ASSEMBLY.md','DESIGN.md','validation-summary.json','removal-sweep.json','adoption-audit.json']:z.write(v/name,name)
 z.writestr('README_FIRST.md',readme+'顔スクリーンは変更していないため、この交換セットには含めていません。旧支持枠から剥がせない場合は従来のface.stlを再印刷してください。\n')
archive=r/'projection-prototype-print.zip';temp=r/'projection-prototype-print-v12.tmp.zip';prefix='DANYA-static-prototype/'
replacements={prefix+str(p.relative_to(r)):p for p in [r/'parts.json',r/'print-validation.json',r/'PRINT_LIST.csv',r/'VERIFIED_FILES.json',r/'ASSEMBLY.md',r/'MAGNET_ASSEMBLY.md',r/'README_FIRST.md']+[r/'print'/(n+'.stl')for n in geo]+[v/n for n in ['covered.png','ASSEMBLY.md','DESIGN.md','validation-summary.json','adoption-audit.json','removal-sweep.json']]}
with zipfile.ZipFile(archive)as old,zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED)as new:
 for info in old.infolist():
  if info.filename not in replacements:new.writestr(info,old.read(info.filename))
 for name,path in replacements.items():new.write(path,name)
temp.replace(archive)
for archive,prefix in [(v/'face-v12-update-print.zip',''),(archive,'DANYA-static-prototype/')]:
 with zipfile.ZipFile(archive)as z:
  assert z.testzip() is None
  for n in geo:assert z.read(prefix+'print/'+n+'.stl')==(r/'print'/(n+'.stl')).read_bytes()
print('Verified update/full ZIPs and print hashes.')
