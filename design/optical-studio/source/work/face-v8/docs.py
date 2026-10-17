from pathlib import Path
import json,numpy as np,zipfile
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'face-v8';q=json.load(open(v/'optical-check.json'));g=json.load(open(v/'geometry-check.json'));s=json.load(open(v/'surface-check.json'))
xyz=np.fromfile('work/torso-flat/face-local-v2.positions.bin','<f4').reshape(-1,3);xyz[:,2]+=8
ids=np.unique(np.fromfile('work/adult-kit/original-face-outer.indices.bin','<u4'));shadow=json.load(open(v/'shadow-points.json'))
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="780" viewBox="0 0 720 780"><rect width="720" height="780" fill="#10212a"/><g fill="#6f8990" opacity=".55">']
def dot(p,color=None,r=.85):return f'<circle cx="{360+p[1]*3:.2f}" cy="{730-(p[2]-40)*3:.2f}" r="{r}"'+(f' fill="{color}"'if color else'')+'/>'
svg += [dot(xyz[i])for i in ids[::12]];svg+=['</g><g fill="#ff715e" opacity=".75">'];svg += [dot(p,r=1.2)for p in shadow];svg+=['</g><g fill="white" font-family="sans-serif"><text x="28" y="36" font-size="23">Projection obstruction / front view</text><text x="28" y="66" font-size="16">Red: blocked samples (not a brightness simulation)</text></g></svg>'];(v/'shadow-map.svg').write_text(''.join(svg))
text=f'''# 顔の増厚・磁石アーム補強 v8（固定式試作）

![顔の内側と補強](reinforced-face.png)

## 変更
- 顔の外面と外側の合わせ縁は維持。投影面の内側へ公称0.5 mmオフセット。細かな内面法線を平滑化し、合わせ縁へ連続させています。元から厚さが一定の顔ではないため、完成品全体を一律の絶対厚さとは表示しません。
- 磁石の戻り腕は、元の約0.4 mmの薄板に対し、0.8 mmのウェブと上下の最大2.0 mmのリブ、高さ20 mmで補強。顔の内側へ回り込む根元を一体化しました。
- 4か所の磁石中心、Ø5.9×2.9 mmの磁石、Ø6.15 mmの穴、接着剤用の内部溝・底の着座部は維持。磁石を二液エポキシで固定する方法も変更しません。
- 交換印刷は `face.stl` 1個。後頭部、鏡、支柱、光学配置は変更なし。既存の組立ガイドに従って取り付けます。顔の印刷では従来どおり造形方向と支持を確認してください。今回、顔のサポート不要を保証する設計ではありません。

## 形状検査
- 閉じた1個の連結体、非多様体エッジ0。
- 元の外面の{q['outerVertexSamples']:,}頂点で最近接面のずれ最大{s['originalOuterSurfaceMaxDeviationMM']:.7f} mm。
- 自己交差検査：隣接面を除く、交差線長0.01 mm超の交差0（数値判定）。
- 補強部と既存部品の面交差：{len(q['reinforcementContacts'])}。顔全体を合わせ面の数値接触除去用に前方0.02 mm移した確認：接触{len(q['faceContactsWith002MMForwardSeamRelief'])}。

## 光路の確認結果
**「影響なし」「全顔100%照射」とは判定していません。顔の縁に磁石アーム由来の遮光が残ります。**

- 検査は外面{q['outerVertexSamples']:,}頂点＋{q['outerTriangleCentroidSamples']:,}三角形重心。既存の内外面混在サンプル数とは異なります。
- 幾何学上の投影経路が作れない点：{q['missingPaths']}。
- 頂点で遮光あり：変更前{q['vertexBefore']:,}、変更後{q['vertexAfter']:,}、新たな遮光{q['vertexNew']:,}。
- 三角形重心を面積で重み付けした遮光面積の概算：変更前{q['centroidEstimatedBlockedAreaBeforeMM2']:.1f} mm²、変更後{q['centroidEstimatedBlockedAreaAfterMM2']:.1f} mm²（検査対象外面積の約{q['centroidEstimatedBlockedPercent']:.3f}%）。輪郭を厳密積分した面積ではありません。
- 後頭部上下、接合座、鏡固定部、支柱、ナット・ねじ等のモデルも含め、顔以外の遮光検出は{sum(n for k,n in q['partHitCounts'].items()if k!='existing-face-holder')}。実体のない参照用頭部・印刷テンプレートは除外。鏡ガラスは意図した反射面として扱います。
- 最初の顔内面への到達を光路終点として比較。顔の凹凸による複数面交差は、変更前{q["multipleScreenIntersectionsBefore"]}点から変更後{q["multipleScreenIntersections"]}点へ増えました。これは別途、実機で確認する必要があります。厚いスクリーン内での光の拡散や屈折は計算していません。

![赤は遮光が検出された点](shadow-map.svg)

検査の生データ：[optical-check.json](optical-check.json)、[geometry-check.json](geometry-check.json)、[surface-check.json](surface-check.json)。この結果は固定式試作の組立位置に限ります。本番の可動機構を検証したものではありません。

## 実機で残る確認
増厚による明るさ・透過率・ぼけの変化、磁石の引抜きと腕の繰返し開閉強度は未試験です。補強は幾何形状の改善であり、強度保証ではありません。実機で均一な全白像と格子を投影して、顔の縁と増厚後の明るさを確認してください。
'''
(v/'README.md').write_text(text)
html='''<!doctype html><meta charset="utf-8"><title>DANYA face v8</title><style>body{background:#10212a;color:#e0ebeb;font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;line-height:1.8}img{max-width:100%;max-height:700px}a{color:#83dace}.warn{padding:20px;background:#4a332d;border-radius:10px}</style><h1>顔の増厚・磁石アーム補強 v8</h1><p>外形を保ち、内側へ公称0.5 mm増厚。0.8 mmウェブ＋最大2 mmの上下リブで磁石アームを補強。</p><img src="reinforced-face.png"><p class="warn">光路の影響はゼロではありません。磁石アームによる顔の縁の影が残ります。変更後の遮光面積の概算は約PERCENT%。明るさ・透過率・実物の強度は未検証です。</p><img src="shadow-map.svg"><p><a href="README.md">寸法・検査結果の詳細</a> / <a href="face-v8-update-print.zip">顔の更新STLセット</a> / <a href="../../../projection-prototype.html?revision=face-v8">組立3Dへ戻る</a></p>'''.replace('PERCENT',f"{q['centroidEstimatedBlockedPercent']:.3f}")
(v/'index.html').write_text(html)
with zipfile.ZipFile(v/'face-v8-update-print.zip','w',zipfile.ZIP_DEFLATED)as z:
 z.write(r/'print/face.stl','print/face.stl')
 for n in ['README.md','reinforced-face.png','shadow-map.svg','geometry-check.json','surface-check.json','optical-check.json']:z.write(v/n,n)
