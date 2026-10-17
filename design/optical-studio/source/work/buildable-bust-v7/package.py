import json,hashlib,zipfile
from pathlib import Path
r=Path('outputs/danya-simulator/assets/buildable-bust-v7');j=lambda n:json.loads((r/n).read_text())
p=j('parts.json');assert len(p['parts'])==42 and all(x['validSolid'] and x['solidCount']==1 for x in p['parts'])
parts=j('parts-check.json');assert len(parts['rows'])==38 and not any(x['hits'] or x['lightBlockedPoses'] for x in parts['rows'])
t=j('transmission-check.json');assert len(t['rows'])==8 and len(t['motorRows'])==3 and not any(x['hits'] for x in t['rows']+t['motorRows']) and not t['mutual'] and not any(x['blockedPoses'] for x in t['lightRows'])
assert not j('assembly-check.json')['rows'] and not j('collar-check.json')['rows']
m=j('motion-check.json');assert not any(x['hits'] for x in m['external']) and all(x['minimumBodySectionMarginMM']>0 for x in m['margins'])
(r/'VERIFICATION.md').write_text('''# v7 検証結果 / 2026-09-23

設計仮定は可動部2kg。うなずき軸Z=-4mm、傾き軸Z=20mm。試験値と計算値を区別する。

| 検査 | 結果 | 範囲 |
|---|---|---|
| CAD形状 | 42部品が有効な単一ソリッド | 本体38＋試験治具4 |
| 本体CAD対頭・顔・胴体・プロジェクタ | 表面交差0 | 38部品、27姿勢 |
| 異なる可動段の支持部品同士 | 表面交差0 | 27姿勢、軸・軸受の意図した接触等を除外 |
| ベルト・プーリー対本体 | 表面交差0 | ベルト2、プーリー包絡6、27姿勢 |
| ベルト・プーリー相互 | 検査対象の交差0 | 同一軸のベルトとプーリーの巻付き接触を除外 |
| モーター | 検査対象との表面交差0 | 実寸ボックス3台、27姿勢 |
| 首カバー表面 | 本体・ベルト・プーリーとの交差0 | 表示用変形モデル、27姿勢 |
| プロジェクタ／頭／顔対胴体 | 表面交差0 | 新しい回転中心で27姿勢 |
| 主光線の遮蔽 | サンプルの遮蔽0 | 顔頂点を200点ごとに間引き、27姿勢 |
| 回転座標の回帰確認 | 既知の90°回転と分離した軸中心を確認 | rigid-obb.test.mjs、motion.test.mjs |
| 画面動作 | 30°/12°/6°とリセットの表示を確認 | ローカルブラウザー |
| 実物強度・クラッチ・耐久 | **未実施 / NOT_RUN** | 治具・手順のみ作成 |

断面モデル（壁厚3mmを差し引く）上のモーター最小余裕は左右16.42mm、うなずき12.12mm、傾き6.07mm。プロジェクタのZ≤24mmの頂点では最小4.96mm。これらは実物表面からの最短距離ではない。

角度は左右{-30,0,30}、前傾{0,6,12}、傾き{-6,0,6}の組合せ。連続軌道の証明、実物公差、熱変形、有限光束、全画素、明るさ・ピント・首カバーの材料ひずみを検証したものではない。

同じ可動段の締結部・全ねじ・ケーブル・未完成の支持金具、完全包含の網羅検査は含まない。プーリーは仮の径・幅の包絡で、購入するハブ・フランジ形状との照合が必要。

**うなずき駆動軸の片持ち曲げは強度確定前。** 詳細設計資料の計算と未完了事項を参照。交差ゼロを、耐荷重や製造可能性の保証として使わない。

根拠JSON：parts-check.json、assembly-check.json、transmission-check.json、collar-check.json、motion-check.json。旧モーター判定の不備はAUDIT_CORRECTION.mdに記録。
''')
# Explicitly label direct-drive adjustment; preserve the solved geometry.
d=j('transmission.json')
for row in d['routes']:
 if row['role']=='pitch':row['motorSlotTravelMM']=row.pop('idlerTravelMM',3)
(r/'transmission.json').write_text(json.dumps(d,indent=2))
files=[r/(x['name']+ext) for x in p['parts'] for ext in ['.step','.stl']]
files += [r/n for n in ['neck-unit-assembly.step','pitch-belt.stl','roll-belt.stl','parts.json','drive-layout.json','transmission.json','spring-selection.json','DETAIL_DESIGN.md','HARDWARE.csv','PHYSICAL_TEST.md','test-results.csv','VERIFICATION.md','AUDIT_CORRECTION.md','parts-check.json','assembly-check.json','transmission-check.json','collar-check.json','motion-check.json']]
assert all(x.exists() for x in files)
manifest={'status':'REVIEW_PROTOTYPE_NOT_MANUFACTURING_RELEASE','date':'2026-09-23','assumedMovingMassKG':2,'physicalTests':'NOT_RUN','unresolved':['pitch shaft bending and pretension','motor and idler brackets','head carrier attachment','purchased pulley hubs/flanges','machining cost within30000JPY'], 'sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}}
(r/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(r/'MANIFEST.json')
out=Path('outputs/DANYA-neck-v7-review.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for x in files:z.write(x,'DANYA-neck-v7/'+x.name)
print('VERIFIED',len(files),'files',out.resolve(),out.stat().st_size)
