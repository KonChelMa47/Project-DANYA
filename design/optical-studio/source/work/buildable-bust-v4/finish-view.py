# -*- coding: utf-8 -*-
from pathlib import Path
p=Path('outputs/danya-simulator/src/budget-system-study.js');s=p.read_text();a=s.index('const a=new T.Vector3(...d.drivenPulleyCenter)');b=s.index('\nfunction bearing',a);s=s[:a]+'}\n'+s[b:]
a=s.index('const grid=');s=s[:a]+'''const beltData=await fetch('assets/buildable-bust-v4/belt-screen.json').then(r=>r.json());for(const span of beltData.rows){const start=new T.Vector3(...span.start),end=new T.Vector3(...span.end),axis=new T.Vector3(...span.axis),z=end.clone().sub(start).normalize(),x=new T.Vector3().crossVectors(axis,z).normalize(),m=new T.Mesh(new T.BoxGeometry(2,6,start.distanceTo(end)),new T.MeshStandardMaterial({color:0x454b52,roughness:1}));m.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(x,axis,z));m.position.copy(start).add(end).multiplyScalar(.5);stages[span.role==='pitch'?1:2].add(m);}
''' +s[a:]
s += "\ndocument.getElementById('interior').onchange({target:document.getElementById('interior')});\n"
p.write_text(s)
p=Path('outputs/danya-simulator/buildable-bust-study.html');s=p.read_text().replace('id="interior" type="checkbox"','id="interior" type="checkbox" checked').replace('ベルト実体・追加の支持柱と軸受・全締結・有限光束・連続動作は未検証です。','ベルト直線部は本体・光路を検査済み。ベルトの巻付き部、追加の支持柱・軸受、全締結、有限光束、連続動作は未検証です。').replace('線はベルトの経路目安です。','濃灰色の帯はベルト直線部です。').replace('<tr><td>サーボ相互／本体</td>','<tr><td>プロジェクタ収納</td><td>10,309姿勢の頂点検査</td></tr><tr><td>サーボ相互／本体</td>');p.write_text(s)
