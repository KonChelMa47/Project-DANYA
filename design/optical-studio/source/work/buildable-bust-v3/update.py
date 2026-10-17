# -*- coding: utf-8 -*-
from pathlib import Path
p=Path('outputs/danya-simulator/src/budget-system-study.js');s=p.read_text().replace("geometry('assets/buildable-bust-v1/bust')","geometry('assets/buildable-bust-v3/body')").replace("geometry(root+'aligned-neck')","geometry('assets/buildable-bust-v3/collar')")
s=s.replace('new T.Mesh(clip(torso,true),chestMat)','new T.Mesh(torso,chestMat)').replace('chest.position.x=drive.torsoShiftX;','chest.position.x=0;').replace('0x7ea0a2','0xdadfdc').replace('0x8eafb5','0xdadfdc')
s=s.replace('(v.z+20)/(neckDatum.sectionZ+20)','(v.z-24)/(neckDatum.sectionZ-24)')
pos=s.index('const headMat=');s=s[:pos]+"const panelMat=new T.MeshStandardMaterial({color:0x303b40,roughness:1,side:T.DoubleSide}),panel=new T.Mesh(await geometry('assets/buildable-bust-v3/panel'),panelMat);scene.add(panel);\n"+s[pos:]
s=s.replace('[chestMat,softMat,headMat]','[chestMat,softMat,headMat,panelMat]').replace('grid.position.z=-182','grid.position.z=-161')
s=s.replace("const ids=['yaw','pitch','roll'];","for(const [id,pos]of Object.entries({front:[800,0,90],side:[0,-800,90],oblique:[630,-740,390]}))document.getElementById(id).onclick=()=>{camera.position.set(...pos);controls.target.set(0,0,40);controls.update();};\nconst ids=['yaw','pitch','roll'];")
p.write_text(s)
p=Path('outputs/danya-simulator/buildable-bust-study.html');s=p.read_text();s=s.replace('頭部に合わせた首の再設計','短い首と滑らかな胸像').replace('ST3215の12V版3台＋軸受支持＋3:1ベルト減速の候補。実寸プロジェクタ・鏡・頭を一体で動かします。','首の真下に胴体を置き直し、肩から胸へつながる輪郭を作り直しました。頭部・顔の形状は元のままです。')
a=s.index('<p class="note">');b=s.index('</p>',a)+4;s=s[:a]+'<p class="note">外観の再設計案です。内部機構の収納・可動域・印刷用の肉厚は再検証中です。</p><p><button id="front">正面</button> <button id="side">側面</button> <button id="oblique">斜め</button></p>'+s[b:]
a=s.index('<h2>首の位置を修正</h2>');b=s.index('<h2>予算の上限枠</h2>',a);s=s[:a]+'''<h2>今回の外観</h2><table><tr><td>肩幅</td><td>300mm</td></tr><tr><td>胴体の高さ</td><td>184mm</td></tr><tr><td>追加の首カバー</td><td>約19mm</td></tr></table><p>長く曲がる首を廃止し、胴体全体の中心を頭の下へ移動。外装を滑らかなオフホワイトで統一し、前面に濃色のサービスパネル案を配置しました。</p><p>Furhatのように顔を引き立てる簡潔な胸像を目指した独自の形状案です。濃色部分は開口・吸排気の完成設計ではありません。</p><p><a href="https://www.furhatrobotics.com/furhat-robot">参考：Furhat公式</a> · <a href="assets/buildable-bust-v3/DESIGN.md">変更内容と未検証事項</a></p><p class="note">スライダーは形状確認用。旧外装の干渉検査結果は適用しません。モーター・光路・配線を含む成立性は未確定です。</p>'''+s[b:]
s=s.replace('灰青：頭の首断面に合わせた柔軟カバー案<br>頭・顔の外形は維持／駆動部は配置を再検討中','首の真下に置いたコンパクトな胸像案<br>頭・顔は元の形状／内部機構は再設計中');p.write_text(s)
