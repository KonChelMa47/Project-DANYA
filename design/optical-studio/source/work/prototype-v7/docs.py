from pathlib import Path
import json,re
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'prototype-v7';m=json.load(open(v/'parts.json'));pts=m['mirrorPolygonMM'];w,h=m['mirrorBoundingSizeMM']
# A physical-size SVG: reflection-side view, no curved cut edges.
poly=' '.join(f'{x+75:.1f},{90-y:.1f}'for x,y in pts)
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="150mm" height="180mm" viewBox="0 0 150 180"><rect x="0.2" y="0.2" width="149.6" height="179.6" fill="#f7f9fa" stroke="#809094" stroke-width="0.2"/><polygon points="{poly}" fill="white" stroke="black" stroke-width="0.25"/><path d="M75 100 L75 82 M72 86 L75 82 L78 86" fill="none" stroke="#297d80" stroke-width="0.4"/><text x="75" y="106" text-anchor="middle" font-size="3">FRONT / +Y</text><text x="75" y="175" text-anchor="middle" font-size="3">DANYA v7 / 1:1 / GLASS 1.5 mm</text>'''
for i,(x,y)in enumerate(pts):svg+=f'<text x="{75+x*.90:.2f}" y="{90-y*.90:.2f}" font-size="3" text-anchor="middle">{i+1}</text>'
svg+='</svg>';(r/'mirror-cutting-outline.svg').write_text(svg);(v/'mirror-cutting-outline.svg').write_text(svg)
text=f'''# 固定式試作 v7：直線カット鏡・スリーブなし・支柱のサポート削減

[変更部品だけの印刷用ZIP](prototype-v7-update-print.zip) ／ [図付き変更ガイド・実寸切り出し図](prototype-v7-guide.pdf)

## 大鏡：直線10辺のガラス

鏡厚はユーザー指定の **1.5 mm**。大鏡の仕上がり外形は **{w:.1f}×{h:.1f} mmの十角形**です。円弧や曲線の切断はありません。元板は **150×180 mm以上**とし、外側の余白を直線で切り落とします。ガラスカッターで切断できる種類のガラス鏡を使用してください。

[実寸SVG](mirror-cutting-outline.svg) は反射面から見た図です。紙に印刷する場合は100%・実寸指定。PDFの50 mm確認線も実測してください。`mirror-cutting-master.stl` は厚さ2 mmの樹脂型で、**外周だけ**が鏡の仕上がり線です。刻印の矢印は向きの目印で、ガラスに穴を開ける指示ではありません。

反射面から見た形で位置を合わせます。鏡の裏面に写す場合は図を左右反転するか、樹脂型を裏返して使います。反射面を傷付けないよう型との間を保護してください。

各頂点の位置（150×180 mm元板の左下を原点、反射面から見た向き）：

|頂点|左端からX mm|下端からY mm|
|---|---:|---:|
'''
for i,(x,y)in enumerate(pts):text+=f'|{i+1}|{x+75:.1f}|{y+90:.1f}|\n'
text+='''
隣り合う頂点を直線で結び、最後は10→1です。切断線の幾何形状を示すもので、材質ごとの切断方法・刃の適合を保証するものではありません。

## スリーブなし：固定角度のミラーユニット

![一体型の受けを備えた大鏡の分解図](../mirror-guide/large-exploded.png)

大小とも固定板に**高さ2.6 mmの硬い受け3か所**を一体化しました。裏板をこの受けへ密着させ、M3×12とワッシャー3組で締めます。弾性スリーブ・ゴムは使いません。架台との接続ねじ2本は従来どおりM3×12です。各鏡に合計5本ずつ使用します。

鏡厚1.5 mmに合わせて裏板・裏板ナット・接着層を0.4 mm後ろへ移し、鏡の反射面位置を保っています。接着層の設計厚は1 mmです。大小の裏板も更新部品なので、旧裏板と混在させません。

今回は角度を所定位置に固定する試作用です。3本の締結ねじを緩めて角度調整する構造ではありません。実物で角度補正が必要な場合は、実測後に受け高さ／専用シムを設計し直します。

大鏡を上側後頭部から取り出す確認経路は、固定ねじを外して鏡の法線方向へ14 mm離し、顔側へ進めながら100 mmにつき10 mm下げる経路です。下側の後頭部と顔を外した状態で、0.5〜1 mm刻みの幾何確認を行っています。無理に押し込まず、上側単体で作業します。

## 左右の大支柱：寝かせて印刷

![支柱の印刷方向](columns-print.png)

左右とも **48×186×22 mm**。付属 `print/` STLの向きのまま、平らな裏面全体を造形台へ付けます。支柱部分の裏面と上下の取付部を同じ面にしたため、胴の下に空中の隙間がありません。

![45度の屋根を持つねじ穴](column-hole-detail.png)

横穴の上部は45度の山形です。M4ねじの軸位置、端面高さ、台座・上部アームへの取付穴位置は維持しました。造形形状の検査では、底面以外に45度を超える下向き面は0 mm²です。この2本は**サポートOFF**、付属方向で印刷します。他の部品のサポート設定とは分けてください。

## 更新する印刷部品（7ファイル）

- `head-column-left.stl` / `head-column-right.stl`
- `printed-mirror-fixed-plate.stl` / `printed-mirror-backing.stl`
- `drop-printed-foldMirror-fixed-plate.stl` / `drop-printed-foldMirror-backing.stl`
- `mirror-cutting-master.stl`（ガラス切り出し用の型）

顔・後頭部の外形、上下分割の接続、磁石、プロジェクタの位置は変えていません。小鏡は長方形のままで、厚さを1.5 mmにしています。

## 検証範囲

- 顔の167,698点すべてで幾何光路が存在し、変更部品による追加の遮蔽なし。
- 変更部品と従来の周辺部品の間に、新たな交差なし（既存の接合面を除く）。
- 更新する印刷用7ファイルは、それぞれ閉じた1つのメッシュ。
- 支柱・固定板は指定方向で45度超の下向き張り出しなし。

**実印刷の成功、強度、ガラス切断の精度、接着保持力、実機での光量・ピントは未試験です。** まず支柱1本とミラー部品を仮組みし、締結と投影を確認してください。
'''
(v/'ASSEMBLY.md').write_text(text)
# Current assembly text.
p=r/'ASSEMBLY.md';s=p.read_text().replace('投影試作 06','投影試作 07');s=re.sub(r'^.*弾性スリーブ.*\n','',s,flags=re.M) if False else s
s=s.replace('長さ3 mmの弾性スリーブ6個','スリーブ不要（v7固定受け）').replace('3 mmスリーブ3個、M3×12調整ねじ3本','一体型受け3か所、M3×12締結ねじ3本').replace('調整ねじ','締結ねじ').replace('調整6','締結6').replace('厚さ1.1','厚さ1.5').replace('129.846×155.403','125.2×158.4').replace('曲線輪郭','直線10辺')
s+='\n## v7の更新部品・印刷方向\n\n[直線カット鏡／スリーブなしの固定受け／サポートなしの支柱](prototype-v7/ASSEMBLY.md)。大小の鏡厚は1.5 mmです。鏡の固定板と裏板、左右の大支柱、大鏡の型を更新してください。\n';p.write_text(s)
p=r/'mirror-guide/ASSEMBLY.md';s=p.read_text().replace('v6','v7').replace('129.846×155.403','125.2×158.4').replace('曲線輪郭','直線10辺').replace('鏡厚1.1 mmを仮定しています','鏡厚はユーザー指定に合わせ1.5 mmです').replace('調整用','締結用').replace('調整ねじ','締結ねじ').replace('鏡の角度を調整する','裏板を固定する').replace('固定板 → 弾性スリーブ → 裏板','固定板の一体型受け → 裏板').replace('3本の締結ねじで傾きを少しずつ合わせます。スリーブの潰れ、','鏡が所定位置に固定されているかを確認します。').replace('ナット・ねじ・鏡・スリーブ','ナット・ねじ・鏡').replace('スリーブの硬さと圧縮量・調整範囲','造形公差・投影品質');s='\n'.join(line for line in s.splitlines()if not line.startswith('|弾性スリーブ'))+'\n\n大小ともスリーブを使わず、固定板と一体の高さ2.6 mmの受けへ裏板を締結します。角度をねじの緩め加減で調整せず、固定した状態で使用してください。\n\n[大鏡の実寸切り出し図・v7変更手順](../prototype-v7/ASSEMBLY.md)\n';p.write_text(s)
p=Path('outputs/danya-simulator/projection-prototype.html');s=p.read_text().replace('STATIC PROTOTYPE 06 / 後頭部2分割・補強接合','STATIC PROTOTYPE 07 / 直線カット鏡・スリーブ不要').replace('revision=split-head-v6','revision=prototype-v7')
s=s.replace('<details open><summary>v6：','<details><summary>v6：');s=s.replace('<details><summary>v6：','<details open><summary>v7：直線カット鏡・スリーブ不要・支柱の印刷</summary><p>大鏡は約125.2×158.4 mmの十角形。鏡厚1.5 mm。大小とも硬い一体型受けで固定し、ゴムは使いません。左右の大支柱は48×186×22 mm、平らな裏面を下にしてサポートOFF。</p><a href="assets/projection-prototype/prototype-v7/prototype-v7-update-print.zip" download>変更7部品のSTLセット</a><br><a href="assets/projection-prototype/prototype-v7/prototype-v7-guide.pdf" target="_blank">変更ガイドと実寸の鏡切り出し図</a><br><a href="assets/projection-prototype/prototype-v7/ASSEMBLY.md">詳しい組立手順</a><p>鏡の角度は固定式。実印刷・強度試験は未実施です。</p></details><details><summary>v6：',1)
p.write_text(s)
