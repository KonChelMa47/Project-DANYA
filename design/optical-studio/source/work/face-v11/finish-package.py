from pathlib import Path
import json,hashlib,zipfile,csv,datetime
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'face-v11';geo=json.loads((v/'release-geometry.json').read_text())
p=r/'PRINT_LIST.csv'
with p.open(encoding='utf-8-sig',newline='')as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
for row in rows:
 name=Path(row['STL']).stem
 if name in geo:
  d=geo[name];row.update({'Width mm':round(d['dimensionsMM'][0],3),'Depth mm':round(d['dimensionsMM'][1],3),'Height mm':round(d['dimensionsMM'][2],3),'A1':'fits','AD5M':'fits (check bed margin)','Role':'v11 detachable face assembly'})
with p.open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
checks={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for folder in ['assembly','print']for p in (r/folder).glob('*.stl')};(r/'VERIFIED_FILES.json').write_text(json.dumps({'timestampUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':'face-v11','scope':'Hashes only, not physical certification. See face-v11/validation-summary.json.','files':checks},indent=2))
(v/'DESIGN.md').write_text('''# v11 着脱構造の見直し\n\nv10では前方へ引き出す途中に上側アーム・磁石が大鏡と後頭部に接触していました。\n\nv11は上側の磁石を大鏡の前側へ移し、長い戻りアームを短縮しました。あご下に剛性のあるつまみを2か所設け、後頭部の下端に抜き出し用の逃げを設けました。顔スクリーンの形状・厚さ、大小鏡、支柱、土台、トレー、光学配置は変更していません。\n\n前方100mmの経路を離散位置で検査し、顔＋支持枠＋顔側磁石と固定部品の接触は0でした。初期15mmは0.25mm間隔、その先は1mm間隔です。任意の傾け方や連続衝突の証明ではありません。\n\n変更した7つの組立メッシュを25万5,542本の公称光路（各3区間）と照合し、変更部品による遮蔽は0でした。印刷する3部品はいずれも1連結・非多様体エッジ0で、規定の閾値を超える面の交差・重複は検出されませんでした。\n\n実物の磁力・接着強度・疲労・造形誤差は未検証です。磁石の吸着力を保証するものではありません。\n\n[組立・着脱手順](ASSEMBLY.md) / [詳細検査結果](validation-summary.json)\n''')
sweep=json.loads((v/'removal-sweep.json').read_text());sweep.pop('stepMM',None);sweep['stepRanges']=[{'fromMM':0,'toMM':15,'stepMM':.25},{'fromMM':15,'toMM':100,'stepMM':1}];(v/'removal-sweep.json').write_text(json.dumps(sweep,indent=2))
readme='最新版はface-v11/ASSEMBLY.mdです。顔ユニットを工具なしで前へ外す構造へ更新。旧ガイドPDF・旧版検査は参考資料です。印刷前にv11手順を優先してください。磁力・接着強度・実物投影は未試験です。\n'
(r/'README_FIRST.md').write_text(readme)
with zipfile.ZipFile(v/'face-v11-update-print.zip','w',zipfile.ZIP_DEFLATED)as z:
 for n in geo:z.write(r/'print'/(n+'.stl'),'print/'+n+'.stl')
 for name in ['ASSEMBLY.md','DESIGN.md','validation-summary.json','removal-sweep.json','adoption-audit.json']:z.write(v/name,name)
 z.writestr('README_FIRST.md',readme+'顔スクリーンは変更していないため、この交換セットには含めていません。旧支持枠から剥がせない場合は従来のface.stlを再印刷してください。\n')
archive=r/'projection-prototype-print.zip';temp=r/'projection-prototype-print-v11.tmp.zip';prefix='DANYA-static-prototype/'
replacements={prefix+str(p.relative_to(r)):p for p in [r/'parts.json',r/'print-validation.json',r/'PRINT_LIST.csv',r/'VERIFIED_FILES.json',r/'ASSEMBLY.md',r/'MAGNET_ASSEMBLY.md',r/'README_FIRST.md']+[r/'print'/(n+'.stl')for n in geo]+[v/n for n in ['ASSEMBLY.md','DESIGN.md','validation-summary.json','adoption-audit.json','removal-sweep.json']]}
with zipfile.ZipFile(archive)as old,zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED)as new:
 for info in old.infolist():
  if info.filename not in replacements:new.writestr(info,old.read(info.filename))
 for name,path in replacements.items():new.write(path,name)
temp.replace(archive)
for archive,prefix in [(v/'face-v11-update-print.zip',''),(archive,'DANYA-static-prototype/')]:
 with zipfile.ZipFile(archive)as z:
  assert z.testzip() is None
  for n in geo:assert z.read(prefix+'print/'+n+'.stl')==(r/'print'/(n+'.stl')).read_bytes()
print('Verified update/full ZIPs and print hashes.')
