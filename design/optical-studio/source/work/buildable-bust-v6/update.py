# -*- coding: utf-8 -*-
from pathlib import Path
p=Path('outputs/danya-simulator/src/budget-system-study.js');s=p.read_text();s=s.replace('assets/buildable-bust-v5/drive-layout.json','assets/buildable-bust-v6/drive-layout.json')
a=s.index('const yawGearLayout=');b=s.index('\n',a)
s=s[:a]+'''const yawGearLayout=await fetch('assets/buildable-bust-v5/gear-layout.json').then(r=>r.json());const yawPinion=new T.Group();yawPinion.position.set(yawGearLayout.pinionCenter[0],yawGearLayout.pinionCenter[1],0);stages[0].add(yawPinion);const detailParts=await fetch('assets/buildable-bust-v6/parts.json').then(r=>r.json());for(const part of detailParts.parts){const geom=await new STLLoader().loadAsync('assets/buildable-bust-v6/'+part.name+'.stl');const metallic=/steel|aluminium/.test(part.material),m=new T.Mesh(geom,new T.MeshStandardMaterial({color:metallic?0xb6c4d5:part.name.includes('friction')?0x4f3931:part.name.includes('liner')?0xe8eeee:part.name.includes('20T')||part.name.includes('96T')?0xd9ad66:0x749ba0,metalness:metallic?.65:.1,roughness:.55}));if(part.parent==='pinion'){yawPinion.add(m);}else{m.position.set(...part.xy,0);stages[part.parent==='yaw'?1:0].add(m);}}''' +s[b:]
a=s.index('const trackShape=');b=s.index('\n',a);s=s[:a]+s[b:]
# Existing bare posts collide with the capture plate: show the output interface only until the upper carrier is revised.
s=s.replace('const post=cylinder(stages[1],[pivot.x,pivot.y+sign*45,0],[0,0,1],4,40,steel);movingSupports.push(post);','')
p.write_text(s)
p=Path('outputs/danya-simulator/buildable-bust-study.html');s=p.read_text().replace('左右回転を歯車駆動へ','左右軸の接続・抜け止め')
s=s.replace('支持部の締結・強度・ケーブル・給電は未確定。','左右軸のホーン接続・保持部をCAD化。強度は条件付きの計算段階です。上部キャリア接続・実機試験・給電は未完了。')
s=s.replace('98／82mm','98／76mm').replace('<td>6mm</td>','<td>10mm（重なり9.5mm）</td>')
a=s.index('<p>1歯分181段階');b=s.index('<h2>首と内部機構',a)
s=s[:a]+'''<p>付属金属ホーンのPCD14・4穴へアダプターを固定。小歯車は独立したM3ボルトとロックナットで保持し、ばね式摩擦クラッチで過負荷を制限します。大歯車は金属保持板・6本の貫通ボルトで抜けを防ぎます。</p><p class="note">試作設計：クラッチ設定0.08±0.01N·m、0.10N·m超は不可。ホーンのねじ規格は現物確認が必要です。歯車が滑った後は原点復帰が必要です。旧支持柱は保持板と競合するため非表示にし、上部キャリア接続は保留しています。</p><p><a href="assets/buildable-bust-v6/DETAIL_DESIGN.md">締結寸法・強度計算・組立手順</a> · <a href="assets/buildable-bust-v6/parts.json">STEP/STL部品一覧</a></p>''' +s[b:]
s=s.replace('リングの穴は最小直径74mm','左右軸の固定スリーブ内径は72mm')
s=s.replace('27姿勢の簡易検査を通過','v6の検査結果は詳細設計資料参照')
p.write_text(s)
