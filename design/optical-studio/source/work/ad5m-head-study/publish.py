import json,zipfile,shutil,hashlib
from pathlib import Path
import numpy as np
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'ad5m-one-plate';layout=json.loads((out/'layout.json').read_text());checks=[];dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
for p in layout['parts']:
 source=r/'print'/(p['part']+'.stl');v=np.fromfile(source,dtype=dt,offset=84)['v'].astype(float);w=np.fromfile(out/(p['part']+'-AD5M.stl'),dtype=dt,offset=84)['v'].astype(float);M=np.array(p['rotation']);t=np.array(p['translation']);err=float(np.max(np.abs(v@M.T+t-w)));assert err<.0001;assert hashlib.sha256(source.read_bytes()).hexdigest()==p['sourceSHA256'];checks.append({'part':p['part'],'triangles':len(v),'maxRigidTransformErrorMM':err,'sourceSHA256':p['sourceSHA256'],'assemblyGeometryChanged':False})
(out/'geometry-verification.json').write_text(json.dumps(checks,indent=2))
readme='''# AD5M：後頭部2パーツを1プレートで印刷

2026-09-25 / v12の形状を保った配置版

## 開くファイル

**DANYA-rear-head-AD5M-settings.3mf** をOrca-Flashforgeで「プロジェクト」として開いてください。上下2個の独立した部品、配置、AD5M 0.4 mm・PLAの仮設定、サポート設定を含みます。印刷前に使用するノズルと材料を確認してスライスしてください。

- 自動配置・自動方向調整・各部品の個別センタリングは行わず、保存した配置を使います。
- 100%倍率、2個を「レイヤーごと」に同時印刷します。「オブジェクトごと」の逐次印刷にはしません。
- 後頭部の形状・分割位置・接合座・磁石穴・ミラー取付部はv12と同じです。印刷方向と配置のみ変更しました。
- DANYA-rear-head-AD5M-plate.3mf は配置だけの汎用3MFです。こちらにはサポート設定がありません。
- DANYA-rear-head-AD5M-plate.stl は配置をまとめたSTLです。2個の離れた部品を含み、くっついていません。STLにはスライス設定が含まれません。
- 個別のhead-upper-AD5M.stlとhead-lower-AD5M.stlは保管・交換用です。個別読み込みで位置がリセットされるスライサーでは、配置済み3MFを使ってください。

## 印刷設定と検証

AD5Mの造形範囲は220×220×220 mm。モデル本体の占有範囲は約207.4×159.1×156.4 mmです。上下を横向きにするため、**サポートが必要**です。

今回確認した条件：Orca-Flashforge 1.4.0、AD5M 0.4 mm、Generic PLA、積層0.20 mm、壁4周、充填15%、通常サポート自動・しきい値30°・全体、上面Z間隔0.20 mm、XY間隔0.35 mm、ブリムなし。

実際のスライス経路について、サポートを含めて造形範囲内であることを確認しました（slice-verification.json）。端部の余裕は約1 mmなので、ブリムやツリーサポートに変更すると範囲外になる可能性があります。変更後は再確認してください。初層は広いサポート面を使います。プレートの清掃と初層の定着を確認します。

スライサー見積りは約15時間23分、PLA約481 g（サポート込み）。材料・速度設定で変わります。実機での印刷成功、反り、支持材の除去性、接合強度の実物検証はまだ行っていません。

## サポートを外して組み立てる

1. プレートが冷えてから上下パーツをそれぞれ取り外します。
2. 大きい開口部と分割面からサポートを少しずつ外します。薄い外殻や磁石受けを支点にしてこじらないでください。
3. ナット挿入口、ボルト通し穴、磁石ポケット、位置決め穴の支持材を取り除きます。横向き印刷なので、穴の内部にも支持材が生成されます。
4. 外面の支持材接触跡と合わせ面のバリを軽く仕上げます。外面に支持痕が付かないことを保証する配置ではありません。
5. v12の組立ガイドに従い、先に大鏡を取り付けてから、位置決めピン2本とM4×20の4本・ナットで上下を接合します。鏡や顔の組立位置は従来通りです。

このプレートは後頭部の上下2個のみです。顔・支持枠・位置決めピンは前のモデルセットを使って別に印刷します。

## 確認図

- plate-view.png：ベッド上の3D配置（支持材を省略した形状図）
- plate-top.png：真上から見た配置
- first-layer.png：実際の初層経路（橙：支持材、青：本体）
- support-footprint.png：全高さの印刷経路を上から重ねた図

## 検証範囲

元STLとの剛体変換一致を確認し、三角形の追加・削除、スケール変更がないことを検証しました。組立状態の形状を変えていないため、この配置変更による光路への新しい影響はありません。従来の光学検証の限界は変わりません。

OrcaのCLIスライスとプロジェクト出力は終了コード0。ヘッドレスのサムネイル生成には描画警告が出るため、配置図を別途同梱しています。実際に印刷した結果ではありません。

公式造形寸法：https://www.flashforge.com/products/adventurer-5m-3d-printer
'''
(out/'README-JA.md').write_text(readme)
html='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DANYA | AD5M 後頭部1プレート</title><style>body{margin:40px auto;max-width:1000px;padding:0 24px;background:#f4f6f8;color:#172d37;font:17px/1.75 system-ui}img{width:100%;border-radius:16px}a{color:#096e83}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.button{display:inline-block;padding:12px 20px;background:#16697b;color:white;border-radius:8px;margin:8px 12px 16px 0}small{color:#526771}@media(max-width:600px){.grid{display:block}}</style><small>PROJECT DANYA / AD5M / v12形状を保持</small><h1>後頭部2パーツを、1枚のプレートで。</h1><p>形状と接合部はそのまま。上下を横向きに配置し、AD5Mの220 mm角に収めました。<strong>サポート付きで印刷</strong>する設定です。</p><a class="button" href="DANYA-rear-head-AD5M.zip" download>モデルと設定 ZIP</a><a class="button" href="DANYA-rear-head-AD5M-settings.3mf" download>設定付き3MF</a><p><a href="README-JA.md">設定・サポート除去・組立の説明</a></p><img src="plate-view.png" alt="上側と下側を横向きに置いた3D配置図"><p>上図は支持材を省略した形状図です。本体約207.4 ×159.1 mm、高さ156.4 mm。上下2個をレイヤーごとに同時印刷します。</p><h2>サポートを含めた印刷経路</h2><div class="grid"><div><img src="first-layer.png"><p>初層。橙は支持材、青は本体。</p></div><div><img src="support-footprint.png"><p>全高さの経路を重ねた図。</p></div></div><p>Orca-Flashforge / AD5M 0.4 mm / PLA / 0.20 mmでスライスし、支持材を含む経路が造形範囲内に収まることを確認しました。推定約15時間23分・481 g。実印刷は未検証です。</p><p>自動配置・自動方向調整はせず、保存した配置を使ってください。横向きのため外面に支持痕が付き、仕上げが必要です。端部に余裕が少ないためブリムや支持材の種類を変更する場合は再確認してください。</p><p><a href="../guide/assembly-guide.pdf?revision=face-v12">v12組立ガイド</a> · <a href="../../../projection-prototype.html?faceProjection=1&revision=face-v12">組立状態へ戻る</a></p></html>'''
(out/'index.html').write_text(html)
main=r.parent.parent/'projection-prototype.html';s=main.read_text();marker='<h1>投影だけを試す骨組み</h1>';addition='<p><a class="button" href="assets/projection-prototype/ad5m-one-plate/index.html" target="_blank">AD5M：後頭部2個を1プレートで印刷</a></p>'
if addition not in s:main.write_text(s.replace(marker,marker+addition))
files=['DANYA-rear-head-AD5M-settings.3mf','DANYA-rear-head-AD5M-plate.3mf','DANYA-rear-head-AD5M-plate.stl','head-upper-AD5M.stl','head-lower-AD5M.stl','README-JA.md','plate-view.png','plate-top.png','first-layer.png','support-footprint.png','layout.json','geometry-verification.json','slice-verification.json']
with zipfile.ZipFile(out/'DANYA-rear-head-AD5M.zip','w',zipfile.ZIP_DEFLATED)as z:
 for f in files:z.write(out/f,f)
 z.write(r/'guide/assembly-guide.pdf','assembly-guide-v12.pdf')
print('Published',out)
