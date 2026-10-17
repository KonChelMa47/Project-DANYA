import json,zipfile,hashlib
from pathlib import Path
import numpy as np
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'head-split-quarter';dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);report={}
for name in ['head-upper','head-lower']:
 p=out/'print'/(name+'.stl');raw=np.fromfile(p,dtype=dt,offset=84)['v'];v,ix=np.unique(raw.reshape(-1,3),axis=0,return_inverse=True);f=ix.reshape(-1,3);edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);e,c=np.unique(edges,axis=0,return_counts=True);parent=list(range(len(v)))
 def root(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 for a,b in e:
  a,b=root(int(a)),root(int(b))
  if a!=b:parent[b]=a
 comps=len(set(root(i)for i in range(len(v))));area=np.linalg.norm(np.cross(raw[:,1].astype(float)-raw[:,0],raw[:,2].astype(float)-raw[:,0]),axis=1);info={'volumeMM3':float(np.einsum('ij,ij->i',raw[:,0].astype(float),np.cross(raw[:,1].astype(float),raw[:,2].astype(float))).sum()/6),'triangles':len(f),'components':comps,'nonTwoFaceEdges':int((c!=2).sum()),'zeroAreaFaces':int((area<1e-10).sum()),'boundsMM':[v.min(0).tolist(),v.max(0).tolist()],'sizeMM':np.ptp(v,axis=0).tolist(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};assert comps==1 and info['nonTwoFaceEdges']==0 and info['zeroAreaFaces']==0,info;report[name]=info
for name in ['head-upper','head-lower']:
 s=json.loads((out/(name+'-surface.json')).read_text());assert s['crossingsOver01MM']==0 and s['coplanarOverlapsOver00001MM2']==0
 ps=json.loads((out/'print'/(name+'-surface.json')).read_text());assert ps['crossingsOver01MM']==0 and ps['coplanarOverlapsOver00001MM2']==0
 rays=json.loads((out/(name+'-rays.json')).read_text());assert rays['blockedSamples']==0
assert not json.loads((out/'hardware-fit.json').read_text())['surfaceContacts']
assert all(not p['blockedSteps']for p in json.loads((out/'nut-entry-check.json').read_text()))
assert not json.loads((out/'face-removal-sweep.json').read_text())['collisions']
report.update({'splitZMM':106.7,'heightRatioUpper':(274.7373046875-106.7)/(274.7373046875-50.650001525878906),'oldCentralFlangesRemoved':True,'opticalSamples':255542,'blockedSamplesBothParts':0,'hardwareSurfaceContacts':0,'physicalPrintAndStrengthTests':False,'samePlateSimultaneousLayoutVerified':False})
(out/'FINAL-VALIDATION.json').write_text(json.dumps(report,indent=2))
# Supersede intermediate pre-repair component counts.
d=json.loads((out/'head-split-check.json').read_text())
for name in ['head-upper','head-lower']:d[name].update({'components':report[name]['components'],'nonManifoldEdges':report[name]['nonTwoFaceEdges'],'volumeMM3':report[name]['volumeMM3'],'finalValidation':'FINAL-VALIDATION.json'})
d['oldCentralJointRemoved']=True;(out/'head-split-check.json').write_text(json.dumps(d,indent=2))
readme='''# 後頭部：上側3/4・下側1/4 分割STL

2026-09-25。高さ比で上側約75%、下側約25%。中央で分割した旧版の上下パーツとは組み合わせません。

## 印刷するもの

- print/head-upper.stl：上側（約142.6 ×159.1 ×168.0 mm）
- print/head-lower.stl：下側（約104.4 ×134.2 ×56.1 mm）
- print/head-alignment-pin-0.stl / head-alignment-pin-1.stl：位置決めピン各1個（従来と同じ形状）

STLの単位はmm。倍率100%。新しい分割面が下になる方向で書き出しています。上下はそれぞれAD5Mの220 mm角の造形範囲に収まります。今回のファイルは分割STLであり、2個同時の1プレート配置ではありません。平らに置いた2個を余裕付きで同時配置する案は確認できていません。

サポートの要否は使用する材料・方向に合わせてスライサーで確認してください。上部の内面と、接合部・ナット穴付近に注意してください。今回は新形状の実機印刷・最終スライスは未検証です。以前の横置き用3MFの設定や配置をそのまま流用しないでください。

## 変更内容

- 分割面を従来のZ165 mmからZ106.7 mmへ移動。
- 旧中央分割位置の4か所のねじ接合座と、その接合座のねじ穴を除去。元の殻とミラー等の取り付け機構を保護して除去しています。
- 新しい合わせ面の左右に各2本、計4本のM4ねじを設置。頭・ワッシャー・ねじ先端の逃げも追加。
- 顔合わせ面の細い縁が上部から分離しないよう、内側に短い接続を追加。顔＋支持枠を前へ外す移動で接触しないことを確認。
- 顔、磁石、プロジェクタトレー、下の土台のモデルは変更しません。

## 上下の接合

必要なもの：M4×20の六角穴付きねじ4本、M4六角ナット4個（対辺7 mm・厚さ3.2 mm）、M4ワッシャー4枚（外径9 mm・厚さ0.5 mm）、位置決めピン2本。

1. 大鏡を上側に取り付けます。ミラーとその固定部品は従来のものを使います。
2. 下側パーツの新しい左右接合座に、各2個のM4ナットを内側の入口から差し込みます。仮保持が必要ならナットの側面だけに少量の接着剤を使い、ねじ山と座面には付けません。
3. 各接合座の中央の穴へ位置決めピンを1本ずつ入れます。中央がピン穴、両隣がM4通し穴です。
4. 上側を合わせます。ワッシャーを付けたM4×20を上から通してナットへねじ込み、4本を交互に少しずつ締めます。プリント品を潰すまで締めないでください。
5. 上下の合わせ面、顔の着脱、大鏡の固定を確認します。

旧中央位置にあるねじ穴を使う手順は、この版には適用しません。assembly/のSTLは組立位置確認用で、印刷にはprint/を使ってください。

## 検証

上下とも連結成分1、非多様体の辺0、面積0の面0。交差長0.01 mm超の自己交差0、所定しきい値を超える面の重なり0。25万5542点・3区間の計算光路に対して上下パーツによる遮蔽0。ねじ・ナット・ピンのモデルとの面交差0、ナットを20 mm差し込む経路で接触0。顔ユニットを前へ0〜100 mm動かす離散検証で後頭部との接触0。

実物での投影、印刷、接合強度、作業工具と指の使いやすさは未検証です。数値の詳細はFINAL-VALIDATION.jsonと付属の検証JSONを参照してください。
'''
(out/'README-JA.md').write_text(readme)
html='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DANYA | 後頭部3/4分割</title><style>body{margin:40px auto;max-width:1050px;padding:0 24px;background:#f4f7f8;color:#19333d;font:17px/1.75 system-ui}img{width:100%;border-radius:16px}a{color:#086d7e}.button{display:inline-block;padding:12px 18px;background:#186c7d;color:white;border-radius:8px;margin:8px 10px 12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:650px){.grid{display:block}}</style><small>PROJECT DANYA / NEW REAR HEAD SPLIT</small><h1>上側3/4、下側1/4。</h1><p>分割線を首元寄りへ移動し、旧中央位置のねじ穴・接合座を除去しました。新しい左右の接合座を計4本のM4ねじと2本の位置決めピンで合わせます。</p><a class="button" href="DANYA-head-upper75-lower25.zip" download>上下STLセット ZIP</a><a class="button" href="print/head-upper.stl" download>上側3/4 STL</a><a class="button" href="print/head-lower.stl" download>下側1/4 STL</a><p><a href="README-JA.md">印刷・組立の説明</a></p><div class="grid"><div><img src="split-view.png"><p>青：上側 約168 mm ／ 茶：下側 約56 mm</p></div><div><img src="assembled.png"><p>組み合わせた状態。顔・ミラー等は省略。</p></div></div><h2>新しい接合部</h2><img src="joint-detail.png"><p>左右の各接合座：両隣の穴はM4ねじ、中央の穴は位置決めピンです。M4×20ねじ4本、ナット4個、外径9 mm・厚さ0.5 mmのワッシャー4枚を使います。</p><p>各STLは新しい分割面を下に、倍率100%で読み込めます。各部品はAD5Mの造形範囲内ですが、このセットは上下2個を同時配置したプレートではありません。サポートとナット穴の処理をスライサーで確認してください。</p><p>閉じた単一メッシュ、自己交差、計算光路、ねじ・ナットの干渉、顔の着脱を検証済み。実機印刷・強度試験は未検証です。</p></html>'''
(out/'index.html').write_text(html)
files=['README-JA.md','FINAL-VALIDATION.json','head-split-check.json','print-layout.json','split-view.png','assembled.png','joint-detail.png','head-upper-rays.json','head-lower-rays.json','head-upper-surface.json','head-lower-surface.json','hardware-fit.json','nut-entry-check.json','face-removal-sweep.json']
files += ['print/'+p.name for p in (out/'print').glob('*.stl')]+['assembly/'+p.name for p in (out/'assembly').glob('*.stl')]
with zipfile.ZipFile(out/'DANYA-head-upper75-lower25.zip','w',zipfile.ZIP_DEFLATED)as z:
 for f in files:z.write(out/f,f)
with zipfile.ZipFile(out/'DANYA-head-upper75-lower25.zip')as z:assert z.testzip()is None
old=r/'ad5m-one-plate/index.html';s=old.read_text();banner='<p style="background:#fff3d6;padding:16px"><strong>新しい分割方式：</strong><a href="../head-split-quarter/index.html">上側3/4・下側1/4のSTLはこちら</a>。旧中央の接合座を取り除いた版です。</p>'
if banner not in s:old.write_text(s.replace('<h1>',banner+'<h1>',1))
main=r.parent.parent/'projection-prototype.html';s=main.read_text();link='<p><a class="button" href="assets/projection-prototype/head-split-quarter/index.html" target="_blank">新しい後頭部：上3/4・下1/4 STL</a></p>'
if link not in s:main.write_text(s.replace('<h1>投影だけを試す骨組み</h1>','<h1>投影だけを試す骨組み</h1>'+link))
print('PUBLISHED',report)
