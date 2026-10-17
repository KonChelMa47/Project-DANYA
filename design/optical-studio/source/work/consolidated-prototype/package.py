import json,zipfile,hashlib,csv,io
from pathlib import Path
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'complete-latest';out.mkdir(exist_ok=True);manifest=json.loads((r/'parts.json').read_text());parts=[p for p in manifest['parts']if p.get('print')];quarter=r/'head-split-quarter';entries=[];hashes=[];overrides={'head-upper','head-lower','head-alignment-pin-0','head-alignment-pin-1'}
for p in parts:
 n=p['name'];source=(quarter if n in overrides else r)/'print'/(n+'.stl');assert source.exists();path='01_PRINT/'+n+'.stl';entries.append((source,path));hashes.append({'file':path,'quantity':1,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size,'source':str(source),'revision':'head-split-quarter'if n in overrides else p.get('revision',p.get('source','current manifest'))})
for n in ['mirror-cutting-master','foldMirror-cutting-master','magnet-fit-coupon','tray-nut-fit-coupon']:
 source=r/'print'/(n+'.stl');path='02_TEMPLATES_AND_COUPONS/'+n+'.stl';entries.append((source,path));hashes.append({'file':path,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size,'source':str(source),'purpose':'template / fit coupon; not an installed robot component'})
docs=[(quarter/'README-JA.md','03_GUIDES/HEAD_NEW_SPLIT.md'),(quarter/'split-view.png','03_GUIDES/head-new-split.png'),(quarter/'joint-detail.png','03_GUIDES/head-new-joint.png'),(r/'face-v12/ASSEMBLY.md','03_GUIDES/FACE_V12.md'),(r/'guide/assembly-guide.pdf','03_GUIDES/ASSEMBLY_V12_REFERENCE.pdf')]
entries+=docs
csvfile=io.StringIO();w=csv.writer(csvfile);w.writerow(['ファイル','印刷個数','採用版']);w.writerows((h['file'],h['quantity'],h['revision'])for h in hashes if 'quantity'in h)
readme='''# DANYA 固定式投影試作機：最新版 全印刷部品セット

2026-09-25。試作機の印刷部品36ファイルと、鏡の切り出し型・確認用パーツ4ファイルをまとめたセットです。モーター付き本番機構のセットではありません。

## 使い方

- 01_PRINT：取り付ける印刷部品36個。原則、各ファイル1個ずつ。単位mm、倍率100%。部品一覧はPARTS.csv。
- 02_TEMPLATES_AND_COUPONS：大鏡・小鏡の切り出し型と、磁石・トレーのはめ合い確認用パーツ。これらは組立部品ではありません。
- 03_GUIDES：組立説明と画像。
- MODEL_MANIFEST.json：採用元、ファイルサイズ、SHA-256。すべて照合済みです。

## このセットで採用した後頭部

上側3/4・下側1/4。旧中央位置のねじ穴・接合座を除去した最新STLです。head-upper.stl、head-lower.stl、位置決めピン2個を含みます。以前の中央分割版や横置き1プレート版は入れていません。

顔と磁石支持枠はv12。トレー、P1〜P4の脚、大小鏡の固定部、支柱、土台は現在の試作機マニフェストの部品を採用しています。ガラス鏡、磁石、ねじ、ナットなどの購入品はSTLとして印刷する部品には含めていません。

## 組立説明の優先順位

**後頭部は03_GUIDES/HEAD_NEW_SPLIT.mdを使ってください。** 同梱のASSEMBLY_V12_REFERENCE.pdfは既存の全体組立ガイドで、後頭部の中央分割・旧ねじ座の図だけは旧仕様です。その部分を新分割の手順に置き換えます。PDFが最新版の新分割図に更新されたという意味ではありません。

顔の磁石は上側が直径3×厚さ1 mm、下側が直径5.9×厚さ2.9 mmです。顔と支持枠は組として着脱します。

## 印刷について

これは各部品のSTLをまとめたZIPです。全36個を1回で印刷するプレート配置、スライス設定、G-codeは含みません。材料・サポート・印刷方向は使用するプリンターと部品に合わせて確認してください。新しい後頭部は各部品がAD5Mの造形範囲に収まりますが、上下2個の同時配置と実機印刷は未検証です。

新しい後頭部のメッシュ・光路・締結部の検証結果は04_HEAD_VALIDATIONに同梱しています。他の部品は現在の採用STLをそのまま収録しました。このZIP作成に伴って全機構の強度試験を新たに行ったものではありません。
'''
# Face guide has relative links to files outside this consolidated bundle; replace with a short self-contained reference.
face=(r/'face-v12/ASSEMBLY.md').read_text();face='元の顔v12の組立手順です。文中の追加資料へのリンクは元プロジェクト用です。全体図は同梱PDF、後頭部はHEAD_NEW_SPLIT.mdを参照してください。\n\n'+face
checks=['FINAL-VALIDATION.json','hardware-fit.json','nut-entry-check.json','face-removal-sweep.json','head-upper-rays.json','head-lower-rays.json']
entries += [(quarter/f,'04_HEAD_VALIDATION/'+f)for f in checks]
zipout=out/'DANYA-prototype-ALL-PARTS-2026-09-25.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
 z.writestr('README_FIRST_JA.md',readme);z.writestr('PARTS.csv','\ufeff'+csvfile.getvalue());z.writestr('MODEL_MANIFEST.json',json.dumps({'date':'2026-09-25','scope':'static projection prototype','installedPrintedParts':36,'templatesAndCoupons':4,'models':hashes},ensure_ascii=False,indent=2))
 for src,path in entries:
  if path.endswith('FACE_V12.md'):z.writestr(path,face)
  else:z.write(src,path)
with zipfile.ZipFile(zipout)as z:
 assert z.testzip()is None
 for h in hashes:assert hashlib.sha256(z.read(h['file'])).hexdigest()==h['sha256']
 assert len([p for p in z.namelist()if p.startswith('01_PRINT/')and p.endswith('.stl')])==36
(out/'README_FIRST_JA.md').write_text(readme);(out/'MODEL_MANIFEST.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2));(out/'PARTS.csv').write_text('\ufeff'+csvfile.getvalue())
link='<p><a class="button" href="../complete-latest/DANYA-prototype-ALL-PARTS-2026-09-25.zip" download>最新版：試作機の全部品 ZIP（36部品＋型4個）</a></p>'
page=quarter/'index.html';s=page.read_text()
if link not in s:page.write_text(s.replace('<h1>上側3/4、下側1/4。</h1>','<h1>上側3/4、下側1/4。</h1>'+link))
print(json.dumps({'zip':str(zipout),'sizeMB':round(zipout.stat().st_size/1e6,1),'printedParts':36,'templates':4,'hashesVerified':40},ensure_ascii=False))
