from pathlib import Path
import json,shutil
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'split-head-v6'
text='''# 後頭部の上下2分割・補強接合 v6

![上下の分割](split-overview.png)

後頭部の外殻そのものを上下2パーツに分割しました。ミラー取付部は上側、首・アーム接続部は下側です。顔の外形、磁石、頭部の外形は従来形状を維持しています。試作機用の変更です。

## 印刷する部品

|ファイル|数量|造形時の外形 約mm|
|---|---:|---|
|head-upper.stl|1|156.8 ×159.1 ×109.7|
|head-lower.stl|1|136.2 ×156.3 ×114.4|
|head-alignment-pin-0.stl|1|直径3.95 ×長さ6|
|head-alignment-pin-1.stl|1|直径3.95 ×長さ6|

![造形方向](print-orientation.png)

付属STLは分割面を造形台へ向け、底面Z=0にしています。mm単位・100%倍率で使用します。後頭部2パーツはA1・AD5Mの造形範囲に収まります。頭頂部など、接合座以外のサポート要否はスライサーで確認してください。

PETGを前提とした試作設計です。接合座・リブは外周を厚くし、局所的に密な充填を使います。初期目安は0.2 mm積層、外周6本、接合部は100%充填。これらは強度を保証する条件ではありません。

旧一体型 `head.stl` と、v5の `mirror-removable-cradle.stl` は今回の印刷セットでは使用しません。組立参照用のheadは印刷用ファイルではありません。トレーはv5の上入れ式を維持します。

## 頑丈にするための接合構造

![接合座・補強リブ](joint-detail.png)

- **M4×20の金属ボルト4本**で締結。位置決めピンは荷重を支える主部品ではありません。
- 接合座の厚さ：下側8 mm、上側10 mm。幅26 mmの接合座を左右各2か所に設けています。
- 各座の両脇に、幅約3.7 mmの補強リブを設けました。
- 内壁へ約42 mmの高さでつなぎ、内側の約2 mmの補強層へ荷重を分散します。
- ナットの上側に約4.5 mmの肩を残し、金属ナットから樹脂へ締結力を伝えます。
- 位置決めピン2本。片方の上側受け穴は長穴にし、造形によるピン間隔のずれを逃がします。

## 金具と組立順

追加する金具はM4×20の六角穴付きボルト4本、M4六角ナット4個（対辺7 mm・厚さ3.2 mm）、M4ワッシャー4枚（外径9 mm・厚さ0.5 mm想定）です。v5の後頭部外側に使っていたM3×16の4組は使用しません。

1. 上下の分割面とピン穴のバリを取り、金具なしで仮合わせします。外周が揃い、強く押さなくても分割面が接することを確認します。
2. 下側のナットは、頭の内側から外壁へ向けて**横にスライド**させ、ねじ穴に合わせます。無理に下から押し込まないでください。組立中の脱落防止にはナット側面だけを少量の適合エポキシで仮固定できます。ねじ穴・上側の荷重を受ける面へ接着剤を入れず、硬化後に組み立てます。接着剤は締結荷重を支えるためのものではありません。
3. 下側へ位置決めピン2本を入れます。両方のピンが底付きせず、上下の面が閉じるための逃げを設けています。
4. 上側の後頭部が単体のうちに、従来の大鏡固定板・裏板・調整ねじ・鏡を取り付けます。鏡の固定板と一体型取付部には従来のM3×12ねじ2本を使用します。鏡ユニットを出し入れする際は、鏡の法線方向（前下方）へ約14 mm離した位置を経由して顔側へ移動する経路を確認しています。
5. 上側をピンへ合わせて下ろします。顔はまだ付けません。上側の内側からワッシャーとM4×20を入れ、4本を交互に少しずつ締めます。工具は3 mmの六角レンチです。短い側をねじ頭へ入れ、長い側を顔の開口方向へ向けます。
6. 樹脂を潰さず、外周のずれ・段差・ガタがないことを確認します。実際の締付けトルクは印刷条件に依存するため未確定です。最後に顔を合わせ、白・格子画像で投影を確認します。

## 検証範囲

後頭部上下はそれぞれ閉じた1つのメッシュです。設計上の合わせ面の隙間は0 mm。元の後頭部外面を20,091本の測定線で比較し、差は数値誤差の範囲です。補強部と金具の追加による幾何光路167,698点の追加遮蔽は0点です。

ナットの横入れ、鏡の出し入れ、レンチを置く空間はCADで確認しました。経路は離散的な確認で、連続した全動作や人の手の空間を保証するものではありません。

**実印刷の嵌合、締結強度、衝撃・繰り返し荷重の試験は未実施です。** 総重量2 kgは従来の仮定であり、耐荷重認証値ではありません。実重量と印刷条件が確定した段階で強度を検証します。
'''
(v/'ASSEMBLY.md').write_text(text)
p=r/'ASSEMBLY.md';s=p.read_text().replace('投影試作 05','投影試作 06').replace('組立部品33点','組立部品35点')
a=s.index('- 後頭部 `head.stl`');b=s.index('\n',a);s=s[:a]+'- 後頭部：`head-upper.stl` 約156.8×159.1×109.7 mm、`head-lower.stl` 約136.2×156.3×114.4 mm。上下ともA1・AD5Mに収まります。分割面を造形台へ向けた付属方向を使用します。'+s[b:]
s=s.replace('|M3×16 ボルト|4|着脱式ミラー取付台と後頭部|','|M4×20 ボルト|4|後頭部上下の補強接合|').replace('|M4 六角ナット|26|上記の接続|','|M4 六角ナット|30|M4×25の接続26＋後頭部上下4|').replace('|52|各接続に2枚。実際の厚さに応じ突出量を確認|','|56|M4×25の接続に52枚＋後頭部上下に4枚|').replace('|26|従来22個＋着脱式ミラー取付台4個|','|22|鏡・トレー・頭部アーム接続|').replace('|34|従来30枚＋着脱式ミラー取付台4枚|','|30|鏡・トレー・頭部アーム接続|')
a=s.index('8. **大鏡**');b=s.index('\n9. **',a);s=s[:a]+'8. **大鏡・後頭部の上下接合**：ミラー取付部は上側の後頭部と一体です。上側単体で大鏡の固定板・裏板・調整ねじ・鏡を組みます。固定板には従来のM3×12の2本を使用。下側へM4ナット4個を横入れし、位置決めピン2本を入れて上下を合わせます。上側からM4×20＋ワッシャー4組で締めます。[v6の補強接合と図解](split-head-v6/ASSEMBLY.md)を参照。'+s[b:]
s=s.replace('(printable-mounts-v5/ASSEMBLY.md)','(printable-mounts-v5/TRAY_ASSEMBLY.md)')
p.write_text(s)
old=r/'printable-mounts-v5/ASSEMBLY.md';tray=old.read_text().split('## ミラーを頭の外で組み立てる')[0];tray=tray[tray.index('## トレーの造形と組立'):];(r/'printable-mounts-v5/TRAY_ASSEMBLY.md').write_text('# 上入れ式トレー v5（継続使用）\n\n'+tray+'\n後頭部は [v6の手順](../split-head-v6/ASSEMBLY.md) を使用します。\n')
# Viewer reads the two shells while keeping old show/hide controls as a group.
p=Path('outputs/danya-simulator/src/projection-prototype.js');s=p.read_text();s=s[:s.index('let mirrorExploded=')]if 'let mirrorExploded='in s else s
s=s.replace("p.name==='head'?0xdce3dd", "['head-upper','head-lower'].includes(p.name)?0xdce3dd")
needle='const points=[]';s=s.replace(needle,"meshes.head=new T.Group();meshes.head.name='head-shell-group';scene.add(meshes.head);meshes.head.add(meshes['head-upper'],meshes['head-lower']);\n"+needle)
s+='''
let headSplit=false;
document.getElementById('head-split-view').onclick=()=>{headSplit=!headSplit;document.getElementById('explode-face').checked=false;exploded();meshes['head-upper'].position.z=headSplit?85:0;for(const[n,m]of Object.entries(meshes)){if(n==='mirror-glass'||n.startsWith('printed-mirror-')||n.startsWith('mirror-head-'))m.position.z=headSplit?85:0;if(n.startsWith('head-seam-bolt-')||n.startsWith('head-seam-washer-'))m.position.z=headSplit?105:0;if(n.endsWith('-head-magnet')){m.geometry.computeBoundingBox();if(m.geometry.boundingBox.getCenter(new T.Vector3()).z>165)m.position.z=headSplit?85:0;}if(n.endsWith('-face-magnet'))m.visible=!headSplit;}meshes.face.visible=!headSplit;document.getElementById('face').checked=!headSplit;document.getElementById('rays').checked=false;rays.visible=false;camera.position.set(420,-420,280);controls.target.set(-20,0,200);controls.update();document.getElementById('head-split-view').textContent=headSplit?'後頭部を組立位置へ戻す':'後頭部の上下分割を見る';};
''';p.write_text(s)
p=Path('outputs/danya-simulator/projection-prototype.html');s=p.read_text().replace('STATIC PROTOTYPE 05 / 上入れナット・着脱ミラー','STATIC PROTOTYPE 06 / 後頭部2分割・補強接合').replace('id="mirror-view">ミラーの着脱を表示','id="head-split-view">後頭部の上下分割を見る').replace('revision=printable-mounts-v5','revision=split-head-v6')
a=s.index('<details open><summary>v5：');b=s.index('<details>',a)
s=s[:a]+'''<details open><summary>v6：後頭部を上下2分割・接合部を補強</summary><p>上側約157×159×110 mm、下側約136×156×114 mm。M4×20の4本と厚い接合座・両側リブ・内壁補強で接合します。位置決めピン2本付き。鏡は上側単体で取り付けられます。</p><a href="assets/projection-prototype/split-head-v6/split-head-update-print.zip" download>後頭部の更新用STLセット</a><br><a href="assets/projection-prototype/split-head-v6/ASSEMBLY.md">補強接合の図解と組立手順</a><p>トレーはv5の上入れ式を継続使用します。</p><a href="assets/projection-prototype/printable-mounts-v5/TRAY_ASSEMBLY.md">トレーの組立手順</a></details>'''+s[b:]
s=s.replace('後頭部は高さ約225 mmのためA1を使用。','後頭部は上下2分割でA1・AD5Mに収まります。').replace('大鏡取付台はv5で着脱式へ変更。','大鏡取付台は上側の後頭部と一体です。').replace('assets/projection-prototype/printable-mounts-v5/ASSEMBLY.md','assets/projection-prototype/split-head-v6/ASSEMBLY.md').replace('v5のねじ・ナットの組立手順','v6の後頭部接合手順')
p.write_text(s)
# Adapt stable render scripts to hide assembly-reference head and render actual halves.
p=Path('work/projection-prototype/render.py');s=p.read_text().replace("bpy.data.objects['head'].hide_render=name=='open-head';", "bpy.data.objects['head'].hide_render=True;\n for part in ['head-upper','head-lower']:bpy.data.objects[part].hide_render=name=='open-head'\n ");p.write_text(s)
p=Path('work/prototype-assembly-guide/render.py');s=p.read_text().replace("list(meshes)","[n for n in meshes if n!='head']").replace("['head','head-arm-left','head-arm-right']","['head-upper','head-lower','head-arm-left','head-arm-right']").replace("['head','face']","['head-upper','head-lower','face']");p.write_text(s)
