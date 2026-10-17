# -*- coding: utf-8 -*-
from pathlib import Path
p=Path('outputs/danya-simulator/src/budget-system-study.js');s=p.read_text();Path('work/buildable-bust-v5/previous-view.js').write_text(s)
s=s.replace("fetch('assets/buildable-bust-v4/selected-drive-layout.json')","fetch('assets/buildable-bust-v5/drive-layout.json')")
s=s.replace('cylinder(stages[i],d.tip,d.driveAxis,8,10,gearMat);','if(i>0)cylinder(stages[i],d.tip,d.driveAxis,8,10,gearMat);')
s=s.replace('for(const part of ringData.parts){','for(const part of ringData.parts.filter(p=>p.stage!==1)){')
a=s.index('// Stationary annular');s=s[:a]+'''const yawGearLayout=await fetch('assets/buildable-bust-v5/gear-layout.json').then(r=>r.json());const yawGears=[];for(const[name,pos,parent]of[['yaw-ring-96T',yawGearLayout.center,stages[1]],['yaw-pinion-20T',yawGearLayout.pinionCenter,stages[0]]]){const g=await new STLLoader().loadAsync('assets/buildable-bust-v5/'+name+'.stl'),m=new T.Mesh(g,new T.MeshStandardMaterial({color:0xd9ad66,metalness:.2,roughness:.6}));m.position.fromArray(pos);parent.add(m);yawGears.push(m);}const yawPinion=yawGears[1];
''' +s[a:]
s=s.replace('yaw.rotation.z=values[0];','yaw.rotation.z=values[0];yawPinion.rotation.z=yawGearLayout.pinionPhaseRad-yawGearLayout.ratio*values[0];')
p.write_text(s)
p=Path('outputs/danya-simulator/buildable-bust-study.html');s=p.read_text();Path('work/buildable-bust-v5/previous.html').write_text(s)
s=s.replace('<h1>短い首の中空3軸機構</h1>','<h1>左右回転を歯車駆動へ</h1>')
a=s.index('<h2>首と内部機構の変更</h2>');s=s[:a]+'''<h2>左右軸の歯車</h2><p>中空の大歯車96歯と小歯車20歯をかみ合わせる構造へ変更。左右スライダーを動かすと、大小の歯車が逆方向に連動します。</p><table><tr><td>減速比</td><td>4.8：1</td></tr><tr><td>歯の規格</td><td>モジュール1／圧力角20°</td></tr><tr><td>軸間距離</td><td>58mm</td></tr><tr><td>大歯車の外径／光路穴</td><td>98／82mm</td></tr><tr><td>歯幅</td><td>6mm</td></tr></table><p>1歯分181段階の歯面交差チェックと、27姿勢の収納チェックを実施。試作歯形であり、実物の摩擦・強度・歯すきは未検証です。小歯車のサーボ接続アダプターと抜け止めは詳細設計が残ります。</p><p><a href="assets/buildable-bust-v5/YAW_GEARS.md">歯車設計の詳細</a></p>'''+s[a:]
s=s.replace('サーボを胸側へ離し、うなずき・傾きをベルトで伝える案です。','左右は歯車駆動。うなずき・傾きは胸側のサーボからベルトで伝える構成です。').replace('金：伝達部の必要空間','金：左右軸の歯車・他軸のプーリー包絡');p.write_text(s)
