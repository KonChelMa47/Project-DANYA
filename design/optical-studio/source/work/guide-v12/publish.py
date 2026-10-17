from pathlib import Path
import json,zipfile,shutil,hashlib
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'face-v12';pdf=Path('output/pdf/DANYA-prototype-assembly-guide.pdf');shutil.copy2(pdf,v/'assembly-guide-v12.pdf')
p=v/'ASSEMBLY.md';s=p.read_text();s=s.replace('\n\nv11の','\n\n[図付き組立ガイド PDF（v12・12ページ）](assembly-guide-v12.pdf)\n\nv11の');p.write_text(s)
p=r/'ASSEMBLY.md';s=p.read_text();s='[図付き組立ガイド PDF（v12・12ページ）](guide/assembly-guide.pdf)\n\n'+s;p.write_text(s)
(r/'README_FIRST.md').write_text('最新版はguide/assembly-guide.pdf（v12・12ページ）とface-v12/ASSEMBLY.mdです。上側磁石3×1mmを4個、下側磁石5.9×2.9mmを4個使用します。旧ミラー専用PDFやv7の資料は機構の参考資料です。小鏡の現行寸法は49×100mmの角落とし形状で、最新ガイドの指示を優先してください。実物の磁力・接着強度・指の届き方は未検証です。\n')
p=Path('outputs/danya-simulator/projection-prototype.html');s=p.read_text().replace('図付き組立ガイド PDF（10ページ）','図付き組立ガイド PDF（v12・12ページ）').replace('href="assets/projection-prototype/guide/assembly-guide.pdf"','href="assets/projection-prototype/guide/assembly-guide.pdf?revision=face-v12"').replace('大小ミラーの3D組立ガイド（4ページ）','旧ミラー機構ガイド（寸法はv12を優先）');p.write_text(s)
for archive,prefix,files in [
 (r/'projection-prototype-print.zip','DANYA-static-prototype/',{str(p.relative_to(r)):p for p in [r/'guide/assembly-guide.pdf',v/'assembly-guide-v12.pdf',v/'ASSEMBLY.md',r/'ASSEMBLY.md',r/'README_FIRST.md']}),
 (v/'face-v12-update-print.zip','',{'assembly-guide-v12.pdf':v/'assembly-guide-v12.pdf','ASSEMBLY.md':v/'ASSEMBLY.md','README_FIRST.md':Path('work/guide-v12/README_UPDATE.md')})]:
 replacements={prefix+n:p for n,p in files.items()};tmp=archive.with_suffix('.tmp.zip')
 with zipfile.ZipFile(archive)as old,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED)as new:
  for info in old.infolist():
   if info.filename not in replacements:new.writestr(info,old.read(info.filename))
  for n,p in replacements.items():new.write(p,n)
 tmp.replace(archive)
 with zipfile.ZipFile(archive)as z:
  assert z.testzip()is None
  for n,p in replacements.items():assert z.read(n)==p.read_bytes()
print('Published guide and both verified ZIPs.')
