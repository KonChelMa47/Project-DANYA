import json
from pathlib import Path
src=Path('outputs/danya-simulator/assets/buildable-bust-v8/parts.json');dst=Path('outputs/danya-simulator/assets/buildable-bust-v9')
rows=[]
for p in json.loads(src.read_text())['parts']:
 n=p['name'];m=p['material'];a='REVIEW';reason='Commercial part dimensions and procurement not yet confirmed.'
 if 'fabricated' in m or 'rod6' in m:a='REDESIGN';reason='Replace fabricated rods/lugs with printed structural frame and standard bolted joints; do not re-label 6mm rods as plastic.'
 elif '6061' in m or 'machined' in m or m=='aluminium pad':a='REDESIGN';reason='Printed structure requires section and fastening redesign; replace taps with accessible captive nuts or through bolts.'
 elif 'purchased' in m:a='RETAIN_CANDIDATE';reason='Verify exact commercial dimensions, availability and load capacity.'
 elif 'PETG' in m:a='REVIEW_PRINT';reason='Check print orientation, retention, load and heat; previous geometric validity is not strength evidence.'
 elif 'spacer' in n or 'standoff' in n:a='STANDARDIZE';reason='Replace bespoke lengths with stocked spacer/shim stack or printed compression-limited spacer; update bolt engagement.'
 elif 'shaft' in n or 'axle' in n:a='STANDARDIZE';reason='Choose commercial shaft/shoulder bolt and shaft collar/retaining hardware; avoid custom cross-drilling.'
 rows.append(dict(name=n,oldMaterial=m,action=a,reason=reason))
(dst/'manufacturing-audit.json').write_text(json.dumps({'manufacturing':'3D printed parts and purchased hardware only; no bespoke metal machining/welding','budgetJPY':100000,'movingMassAssumptionKg':2,'status':'redesign in progress; not released','parts':rows},ensure_ascii=False,indent=2))
lines=['# v9 製法変更・未完了項目','', '製法：3Dプリント＋市販金具。予算上限：100,000円。現在のST3215モーターを継続使用。可動部質量：仮定2kg。','', 'v8全体は製作承認済みではありません。v9のモータークランプ候補は部分設計で、フレームへの接続と全体の干渉検証は未完了です。','', '| 部品 | 対応 |','|---|---|']
lines += [f"| {r['name']} | {r['action']} |" for r in rows]
lines += ['', '## 次の設計作業', '', '1. モータークランプを樹脂のねじ切りからM3貫通ボルトとナットへ変更。3位置の候補をCADで比較する。', '2. 製作棒フレームを廃止。鏡・頭部荷重を受けるプリント骨格と、プロジェクタの支持骨格を設計する。既存6mm金属棒の樹脂化は禁止。', '3. 首支持環・軸受保持部・床板をプリント向け断面に再設計し、ナットの挿入方向と工具のアクセスを確保する。', '4. 特注軸・スペーサー・プーリーを市販寸法へ統一する。', '5. 全締結部、可動域、光路、ケーブル接続、予算を再評価する。', '6. 2kgの静荷重・繰返し動作・クランプ温度試験を実物で行う。現時点で実物試験は実施していない。', '', '鏡の外形と市販サイズの両立も未確定。現行の切り落とした鏡形状を、市販の長方形鏡として購入できるとは扱わない。', '依存ライブラリの追加インストール、購入、外部への発注は行っていない。']
(dst/'MANUFACTURING.md').write_text('\n'.join(lines)+'\n')
print(len(rows),'parts audited')
