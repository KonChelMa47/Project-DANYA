# -*- coding: utf-8 -*-
from pathlib import Path
p=Path('outputs/danya-simulator/src/budget-system-study.js');s=p.read_text();Path('work/buildable-bust-v4/previous-view.js').write_text(s)
s=s.replace("fetch(root+'selected-drive-layout.json')","fetch('assets/buildable-bust-v4/selected-drive-layout.json')").replace('const pivot=new T.Vector3(...neckDatum.center)','const pivot=new T.Vector3(...drive.pivot)')
s=s.replace('const stages=[new T.Group(),new T.Group(),new T.Group()];for(const stage of stages){scene.add(stage);stage.visible=false;} /* Retired actuator layout: static reference only. */','const stages=[new T.Group(),new T.Group(),new T.Group()];scene.add(stages[0]);yaw.add(stages[1]);pitch.add(stages[2]);stages[1].position.copy(pivot).negate();stages[2].position.copy(pivot).negate();')
s=s.replace('cylinder(stages[i],d.drivenPulleyCenter,d.driveAxis,22,10,gearMat);','if(i>0)cylinder(stages[i],d.drivenPulleyCenter,d.driveAxis,22,10,gearMat);')
s=s.replace('const pts=[a.clone().addScaledVector(side,19.1)','const pts=[a.clone().addScaledVector(side,19.1)')
s=s.replace('stages[i].add(line);','if(i>0)stages[i].add(line);')
a=s.index('for(const z of[-163,-148])');b=s.index('const grid=',a)
s=s[:a]+'''const mechanism=new T.Group();scene.add(mechanism);const movingSupports=[];
for(const sign of[-1,1]){bearing(stages[1],[pivot.x,pivot.y+sign*48,20],[0,1,0]);bearing(stages[2],[pivot.x+sign*46,pivot.y,20],[1,0,0]);const post=cylinder(stages[1],[pivot.x,pivot.y+sign*48,0],[0,0,1],4,40,steel);movingSupports.push(post);}
const ringData=await fetch('assets/buildable-bust-v4/rings.json').then(r=>r.json());const ringMeshes=[];for(const part of ringData.parts){const geom=await new STLLoader().loadAsync('assets/buildable-bust-v4/'+part.name+'.stl'),m=new T.Mesh(geom,new T.MeshStandardMaterial({color:part.stage===3?0xc3b18b:0x749ba0,metalness:.3,roughness:.55}));const parent=[null,stages[1],stages[2],rig][part.stage];parent.add(m);ringMeshes.push(m);}
// Stationary annular sliding track envelope; seats, gear teeth and fasteners are not yet detailed.
const trackShape=new T.Shape();trackShape.absarc(0,0,52,0,2*Math.PI,false);const trackHole=new T.Path();trackHole.absarc(0,0,40,0,2*Math.PI,true);trackShape.holes.push(trackHole);const track=new T.Mesh(new T.ExtrudeGeometry(trackShape,{depth:3,bevelEnabled:false,curveSegments:80}).translate(pivot.x,pivot.y,-26),steel);stages[0].add(track);
''' +s[b:]
s=s.replace("document.getElementById('drives').onchange=e=>stages.forEach(s=>s.visible=e.target.checked);","document.getElementById('drives').onchange=e=>{stages.forEach(s=>s.visible=e.target.checked);ringMeshes.forEach(s=>s.visible=e.target.checked);};")
p.write_text(s)
p=Path('outputs/danya-simulator/buildable-bust-study.html');s=p.read_text();Path('work/buildable-bust-v4/previous.html').write_text(s)
s=s.replace('<h1>短い首と滑らかな胸像</h1>','<h1>短い首の中空3軸機構</h1>').replace('外観の再設計案です。内部機構の収納・可動域・印刷用の肉厚は再検証中です。','外観を維持して内部機構を再配置しました。リング・サーボ・軸受の配置案です。支持部の締結・強度・ケーブル・給電は未確定。')
s=s.replace('id="yaw" type="range" min="-15" max="15"','id="yaw" type="range" min="-30" max="30"').replace('id="pitch" type="range" min="-5" max="5"','id="pitch" type="range" min="0" max="12"').replace('id="roll" type="range" min="-5" max="5"','id="roll" type="range" min="-6" max="6"').replace('旧駆動配置を参考表示（再設計が必要）','新しい内部機構を表示').replace('id="drives" type="checkbox"','id="drives" type="checkbox" checked')
a=s.index('<h2>今回の外観</h2>');s=s[:a]+'''<h2>首と内部機構の変更</h2><p>首の前後位置は維持。回転中心は接続面から22.6mm下の内側へ置き、中空リングで光路を確保します。サーボを胸側へ離し、うなずき・傾きをベルトで伝える案です。</p><table><tr><td>左右の確認範囲</td><td>±30°</td></tr><tr><td>うなずきの確認範囲</td><td>正面～前傾12°</td></tr><tr><td>傾きの確認範囲</td><td>±6°</td></tr><tr><td>サーボ相互／本体</td><td>27姿勢の簡易検査を通過</td></tr><tr><td>頭・肩／中空リング</td><td>27姿勢で表面交差なし</td></tr></table><p>後傾はプロジェクタが胸へ張り出すため未確保です。この範囲も実機保証ではありません。リングの穴は最小直径82mm。金色はプーリーの必要空間、線はベルトの経路目安です。</p><p><a href="assets/buildable-bust-v4/MECHANISM.md">機構設計・検査結果と未解決点</a></p>'''+s[a:]
s=s.replace('スライダーは形状確認用。旧外装の干渉検査結果は適用しません。モーター・光路・配線を含む成立性は未確定です。','検査は指定サンプル姿勢の幾何学検査です。ベルト実体・追加の支持柱と軸受・全締結・有限光束・連続動作は未検証です。')
s=s.replace('首の真下に置いたコンパクトな胸像案<br>頭・顔は元の形状／内部機構は再設計中','青緑：中空支持リング　黒：実寸サーボ<br>金：伝達部の必要空間　銀：軸受・支持の配置案');p.write_text(s)
