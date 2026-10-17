from pathlib import Path
import shutil,json
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'printable-mounts-v5'
text='''# 固定式投影試作 v5 — 上入れナット・着脱式ミラー

## 更新する印刷部品

|ファイル|数|変更|
|---|---:|---|
|drop-projector-tray-t5.stl|1|底から立ち上がるナット座、上入れの六角穴|
|tray-nut-cover-0〜3.stl|各1|抜け止め兼プロジェクタ支持シュー。4個同形状|
|tray-nut-fit-coupon.stl|1|トレーを印刷する前の嵌合確認用。組み込まない|
|head.stl|1|ミラー取付台を分離。外側から締めるねじ穴4か所|
|mirror-removable-cradle.stl|1|鏡を頭の外で組み立てる着脱式取付台|

旧トレー・旧カバーとv5部品は混在させません。後頭部も再印刷が必要です。顔・磁石・左右の支柱とアーム・鏡・鏡の固定板／裏板はそのまま使用します。本番のモーター機構は今回の更新対象に含みません。

## ねじの変更

- トレーと架台：M3×10を4本 → **M3×12を4本**に交換。既存のM3ナット・ワッシャーを使用。
- 着脱式ミラー取付台：**M3×16を4本、M3ナット4個、M3ワッシャー4枚**を追加。
- ワッシャーは外径7 mm・厚さ0.5 mm、ナットは対辺5.5 mm・厚さ2.4 mm想定。ねじ長は頭の下から測ります。

## トレーの造形と組立

![ナット座と別体シュー](tray-pocket-detail.png)
![トレー上側の分解図](tray-top-load.png)

図では部品を浮かせて内部を示しています。**印刷はZIPのprint/内のSTLを使い、分解図の配置では印刷しません。**

1. PETGで確認用パーツとシュー1枚を印刷し、ナットの収まり・爪の保持を確認します。STLはmm単位・100%、付属の造形方向を使用。
2. トレーは平らな下面を造形台へ向けます。ナット穴は上向きに開口し、穴の底には造形台から続く厚さ2.35 mmの床があります。六角穴を支えるサポートは不要な形状です。スライサーの断面で床から穴壁が連続していることを確認してください。
3. シューも付属の平らな下面を下にして別々に印刷します。ガイドの斜面は45°。ナット穴をサポートで埋めないでください。別部品や他の場所のサポート要否とは区別します。
4. トレー上側から4個のナットを入れます。中央の通風窓側から外周へシューをスライドさせます。必要なら左右の爪を少し内側へ押し、受けのくぼみにロックします。
5. シューが戻らずナットが落ちないことを確認して、架台側からM3×12＋ワッシャーで締結します。ねじはシュー内の逃げ穴へ入ります。長いねじへ置き換えないでください。
6. プロジェクタの135×75 mmの底面を受けに載せ、既存の4個のクリップで固定します。支持面の高さと光学位置は変更していません。シュー中央の上面が支持面です。

## ミラーを頭の外で組み立てる

![着脱式ミラーユニット](mirror-cartridge.png)
![後頭部外側のねじアクセス](rear-access.png)

1. 新しい取付台の鏡側から、深い六角穴4か所にM3ナットを入れ、奥の座面へ着座させます。鏡を組む前に行います。扱う間の脱落が心配な場合は、ナット側面だけを少量の適合エポキシで固定し、完全硬化させます。ねじ穴・ナット下の座面へ接着剤を入れません。
2. 頭の外で、取付台へ既存の大鏡固定板を既存のM3×12ねじ2本とワッシャーで固定します。裏板・調整ねじ・弾性スリーブ・鏡を従来どおり組みます。反射面に触れず、鏡を3点接着で保持します。
3. 顔を外し、鏡ユニットを顔側の開口から入れます。最終位置から鏡の法線方向（前下方）へ約20 mm離れた位置を経由し、取付面へ戻します。取り外しは逆順です。まっすぐ顔方向だけに動かすと上側に当たるため、先に取付面から離します。
4. 鏡ユニットを手で支え、後頭部の外側の4か所からM3×16と厚さ0.5 mmのワッシャーで固定します。対角順に少しずつ締め、取付面が接することを確認します。樹脂を割るほど締めません。
5. 白画像・格子画像で投影を再確認してから顔を取り付けます。位置は設計上維持していますが、造形公差・締結によるずれは実機調整が必要です。

後頭部の外殻そのものを二分割した構造ではありません。鏡の取付台を別部品にし、後頭部外側にねじ用のくぼみを4か所追加しています。

## 確認した範囲

- ナット座：床の欠落・上方の挿入を妨げる形状はCAD上で0。爪以外のスライド経路、4本の架台との食い込みも0。
- ミラー：外側ワッシャー座面とナット受け面を確認。締結荷重は取付台の肩を介して伝わる構造。
- 着脱経路：鏡・取付台・関連金具を0.5 mm間隔で移動し、後頭部との接触なし。連続角度検証や実物の挿入試験ではありません。
- 光路：167,698点の幾何光路で、今回の変更による追加遮蔽は0点。全顔の光量・焦点を保証する試験ではありません。
- 実印刷での嵌合、爪の保持力、締結強度、放熱、投影品質は未試験です。
'''
(v/'ASSEMBLY.md').write_text(text)
p=r/'ASSEMBLY.md';s=p.read_text().replace('投影試作 04','投影試作 05').replace('9ページ','10ページ').replace('組立部品32点','組立部品33点')
s=s.replace('|M3×10 ボルト|4|プロジェクタ受けと4本の架台|','|M3×12 ボルト|4|プロジェクタ受けと4本の架台（旧M3×10から変更）|\n|M3×16 ボルト|4|着脱式ミラー取付台と後頭部|').replace('|22|従来18個＋後頭部接続4個|','|26|従来22個＋着脱式ミラー取付台4個|').replace('|30|従来18枚＋後頭部接続に各3枚、計12枚|','|34|従来30枚＋着脱式ミラー取付台4枚|').replace('各架台の上端にM3×10','各架台の上端にM3×12')
start=s.index('4. **プロジェクタ受け**');end=s.index('\n5. **',start)
s=s[:start]+'4. **プロジェクタ受け**：上入れ式ナットと別体シューへ変更しました。上からナットを入れ、シュー4枚を通風窓側から外周へ差して爪をロックします。架台からM3×12で締めます。旧トレー・カバーは再印刷が必要です。[v5の図解手順](printable-mounts-v5/ASSEMBLY.md)を参照。受け開口135.8×75.8 mmと支持高さは維持しています。'+s[end:]
start=s.index('8. **大鏡**');end=s.index('\n9. **',start)
s=s[:start]+'8. **大鏡**：`mirror-removable-cradle.stl` を別印刷します。鏡側からM3ナット4個を入れ、頭の外で大鏡の固定板・裏板・調整ねじ・鏡を組み立てます。既存のM3×12の2本は固定板と取付台に使用。顔側の開口からユニットを入れ、後頭部外側から追加のM3×16＋ワッシャー4組で固定します。[着脱経路と組立順](printable-mounts-v5/ASSEMBLY.md)を確認してください。旧一体型後頭部は再印刷が必要です。'+s[end:]
p.write_text(s)
p=Path('outputs/danya-simulator/projection-prototype.html');s=p.read_text().replace('STATIC PROTOTYPE 04 / トレーナット抜け止め','STATIC PROTOTYPE 05 / 上入れナット・着脱ミラー').replace('9ページ','10ページ').replace('revision=tray-nut-retention-v4','revision=printable-mounts-v5')
a=s.index('<details open><summary>トレーのナット抜け止め');b=s.index('<details>',a)
s=s[:a]+'''<details open><summary>v5：印刷しやすいナット座・着脱式ミラー</summary><p>トレーの六角穴は床から立ち上がる上入れ式。別体シューが抜け止めとプロジェクタ支持を兼ねます。架台ねじはM3×12へ変更。鏡の取付台は別部品になり、後頭部の外からM3×16の4本で固定します。</p><a href="assets/projection-prototype/printable-mounts-v5/mounts-update-print.zip" download>更新用STLセット v5</a><br><a href="assets/projection-prototype/printable-mounts-v5/ASSEMBLY.md">図解・印刷方向・ねじの変更</a></details>'''+s[b:]
s=s.replace('大鏡の取付台と4か所のM3ナット受けを後頭部へ一体化。','4か所のアーム用M3ナット受けは後頭部と一体。大鏡取付台はv5で着脱式へ変更。')
s=s.replace('<button id="head-connection-view">','<button id="mirror-view">ミラーの着脱を表示</button><button id="head-connection-view">')
s=s.replace('assets/head-connectors-v3/FASTENER_GUIDE.md','assets/projection-prototype/printable-mounts-v5/ASSEMBLY.md').replace('ねじ・金属ナットの入れ方（試作／本番）','v5のねじ・ナットの組立手順')
p.write_text(s)
p=Path('outputs/danya-simulator/src/projection-prototype.js');s=p.read_text();s+='''
let mirrorExploded=false;
document.getElementById('mirror-view').onclick=()=>{mirrorExploded=!mirrorExploded;for(const [n,m]of Object.entries(meshes)){if(n==='mirror-removable-cradle'||n==='mirror-glass'||n.startsWith('printed-mirror-')||n.startsWith('mirror-head-')||n.startsWith('mirror-cartridge-nut-'))m.position.set(mirrorExploded?109.9469:0,mirrorExploded?-.2933:0,mirrorExploded?-17.3486:0);}document.getElementById('face').checked=!mirrorExploded;meshes.face.visible=!mirrorExploded;for(let i=0;i<4;i++)meshes['magnet-clear-'+i+'-face-magnet'].visible=!mirrorExploded;document.getElementById('rays').checked=false;rays.visible=false;camera.position.set(400,-330,250);controls.target.set(20,0,185);controls.update();document.getElementById('mirror-view').textContent=mirrorExploded?'ミラーを組立位置へ戻す':'ミラーの着脱を表示';};
''';p.write_text(s)
# Revise the stable illustrated guide.
p=Path('work/prototype-assembly-guide/build-pdf.py');s=p.read_text().replace('投影試作04','投影試作05').replace('STATIC PROTOTYPE 04','STATIC PROTOTYPE 05').replace('tray-nut-retention-v4 / 2026-09-23','printable-mounts-v5 / 2026-09-24').replace('{page} / 9','{page} / 10').replace('組立部品32点','組立部品33点').replace('M3×10','M3×12')
s=s.replace("['小鏡架台1本・大小の固定板と裏板','5'","['小鏡架台・固定板・裏板・着脱式取付台','6'")
s=s.replace("['M3×8','4'","['M3×16','4','着脱式ミラー取付台と後頭部'],['M3×8','4'").replace("'22 / 30'","'26 / 34'")
s=s.replace('改訂03の後頭部・左右アームと、固定式試作の印刷セットです。','改訂05のトレー・着脱式ミラー取付台・後頭部を含む固定式試作です。')
s=s.replace('の裏面4か所へM3ナットを先に入れ、カバー4枚を横から差してロックします（9ページ参照）。','の上側4か所へM3ナットを入れ、支持シュー4枚を横から差してロックします（9ページ参照）。')
s=s.replace('大鏡は次ページの頭部作業後、後頭部の一体型取付台へ固定します。','大鏡は頭の外で着脱式取付台へ組みます（10ページ参照）。')
s=s.replace('4. 大鏡の固定板を後頭部の一体型取付台へ、M3×12＋ワッシャー2組で固定します。旧 mirror-head-cradle を別付けする工程はありません。','4. 大鏡は着脱式取付台へM3×12＋ワッシャー2組で固定します。頭部への取り付けには別途M3×16を4本使用します（10ページ）。')
start=s.index("start('08 / トレー")
s=s[:start]+'''start('08 / 上入れ式ナットと支持シュー','改訂05：トレーとシュー4枚を再印刷。架台のねじ4本をM3×12へ変更。')
img('tray-pocket-detail.png',44,405,507,290)
y=p('分解図：青緑＝受け、銀＝ナット、橙＝抜け止め兼支持シュー。',391,10)
y=p('1. 確認用パーツとシュー1枚をPETGで先に印刷。付属のprint/のSTLをそのままの方向で使います。図の浮いた配置では印刷しません。',y,11)
y=p('2. 六角穴は上向き。穴の底は造形台から続く厚さ2.35 mmの床で支えています。ナット穴内のサポートを必要としない形状です。',y,11)
y=p('3. 上からナットを入れ、通風窓側から外周へシューを差します。左右の爪をロックし、戻らずナットが落ちないことを確認します。',y,11)
y=p('4. 架台側からM3×12＋ワッシャーで締結。プロジェクタをシューの上面へ載せ、従来のクリップ4個で固定します。',y,11)
p('旧トレー・旧カバーは混在不可。支持面の高さと投影位置は維持しています。造形公差、爪の保持力、締結強度・温度は実物試験前です。',y,10)
finish()
start('09 / ミラーをユニットで着脱','追加：M3×16を4本、M3ナット4個、外径7×厚さ0.5 mmのワッシャー4枚。')
img('mirror-cartridge.png',44,433,308,257);img('rear-access.png',362,433,189,257)
y=p('左：鏡と取付台を外した状態。右：後頭部の外側からねじを締めます。',422,10)
y=p('1. 取付台の鏡側から深い六角穴4か所へナットを入れ、奥の座面に着座させます。必要なら側面だけ少量の適合エポキシで保持。ねじ穴・座面には入れません。',y,11)
y=p('2. 頭の外で、既存のM3×12ねじ2本により大鏡の固定板を取付台へ固定。裏板・調整ねじ・鏡を従来どおり組み立てます。',y,11)
y=p('3. 顔側の開口から入れます。最終位置から鏡の法線方向（前下方）へ約20 mm離れた位置を経由して着座。外す時は先に20 mm離し、それから顔側へ引き出します。',y,11)
y=p('4. ユニットを支え、後頭部の外側からM3×16＋ワッシャー4組で固定。対角順に少しずつ締め、白・格子画像で投影を再確認します。',y,11)
p('後頭部は再印刷が必要です。外殻の二分割ではなく取付台の着脱式です。0.5 mm刻みの経路確認で後頭部との接触なし。実物の挿入・強度試験は未実施です。',y,10)
finish();c.save();shutil.copy2(out,a/'guide/assembly-guide.pdf');print(out)
'''
for n in ['tray-pocket-detail','mirror-cartridge','rear-access']:shutil.copy2(v/(n+'.png'),r/'guide'/(n+'.png'))
p.write_text(s)
