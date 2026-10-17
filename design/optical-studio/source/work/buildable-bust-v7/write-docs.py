# -*- coding: utf-8 -*-
from pathlib import Path
import json,csv
r=Path('outputs/danya-simulator/assets/buildable-bust-v7')
spring={'manufacturer':'昌和発條製作所','partNumber':'AP080-022-1.2','oldPartNumber':'A08-05','material':'SWP-A','outerDiameterMM':8,'wireDiameterMM':1.2,'innerDiameterNominalMM':5.6,'freeLengthMM':22,'springRateNPerMM':6.803,'maximumDeflectionMM':7,'maximumLoadN':47.36,'totalTurns':11.5,'designInstalledLengthMM':[16,18],'nominalPreloadN':[27.212,40.818],'clutchInputSlipTargetNm':.08,'clutchInputAcceptableNm':[.07,.09],'rejectAboveNm':.10,'guideOuterDiameterMM':5.2,'guideLengthMM':14.5,'retainingBolt':'M3x40','source':'https://showa-spring.com/wp-content/themes/showa/images/page/catalog.pdf','secondarySource':'https://jp.c.misumi-ec.com/book/VONA_ME01/pdf/0520.pdf','purchaseStatus':'Not ordered; purchase SKU chosen for prototype. Friction calibration not performed.'}
(r/'spring-selection.json').write_text(json.dumps(spring,ensure_ascii=False,indent=2))
(r/'PHYSICAL_TEST.md').write_text('''# 実物試験手順 — まだ未実施

2026-09-22。ユーザー了承の可動部2kgを設計仮定とする。CAD・計算結果を実物試験の合格として扱わない。`test-results.csv` は実測して埋める。

## 試験対象と治具

- 頭・プロジェクタを外した首ユニットと、選定した実物のねじ・軸受・ばね。
- `test-base-adapter`：180×160×8、6穴PCD108、四隅M6穴。アルミ板。ベンチへM6と金属スペーサー20mm以上で固定し、下面のねじ頭の空間を確保。
- `test-interface-plate`：出力側4穴を使う負荷取付板。支持架台を外して左右軸単体を試験する時に用いる。
- `test-load-lever`：50mm間隔の負荷取付穴を持つレバー。試験接続は回り止めを含む2点固定とし、長さは回転中心から測る。
- `clutch-test-lever`：外径22mmの小歯車を周囲から保持する割りクランプ。力を掛ける穴は軸中心から50mm。
- フォースゲージ（0.1N以下の分解能）、ノギス／ダイヤルゲージ、秤、既知の錘。これらを遠隔で操作した記録はない。

治具は試験用で、ロボットの内部には取り付けない。主構造がアルミ指定の箇所をPETGに置き換えた場合、この荷重条件の承認にはならない。

## 1. クラッチ校正

1. 圧縮ばねAP080-022-1.2を入れ、初期の組付け長を18mmにする。ガイドの底付き、ばね密着がないことを確認。
2. アダプター外周を軟質口金の治具で固定し、歯車へ測定レバーを取り付ける。サーボを通電ストールさせて固定する方法は使わない。
3. 軸を垂直に置き、50mm穴を**水平・接線方向**へゆっくり引く。上下方向へ引くだけでは左右軸のトルクを測れない。
4. 滑り始めの力1.4～1.8Nを合格範囲とする（0.07～0.09N·m）。2.0Nを超えたら設定を戻す。CW/CCW各5回を記録。
5. ナット位置とばね長を記録し、慣らし50回後に再測定。組付け長16mmを下回る締め込みは行わず、範囲内で目標に合わなければ摩擦材を再選定する。
6. 錘だけを使う場合は軸を水平にしてレバー自重を差し引く。単に縦軸レバーへ錘を吊るす方法では測れない。

滑った後の頭の角度はサーボ内エンコーダと一致しない。原点合わせをやり直す。自動原点復帰制御は別途実装する。

## 2. 左右軸の保持部

1. 固定板を試験台へ固定。試験時は板厚追加に合わせM3×35等でねじ長さを合わせ、ロックナットの全ねじ山が掛かることを確認。ロボット用M3×25をそのまま流用しない。
2. 20N、40N、60Nへ段階的に軸方向荷重を増やし、各60秒保持。荷重を外し、10分後の残留変形を測る。
3. 反対方向も試験する。下向きに載せた錘だけでは上側抜け止めの引抜き試験にはならない。ユニットを反転した治具配置で上側保持を検証する。
4. 回転中心から50mmで60Nを与え、軸力60N＋転倒モーメント3N·mの組合せを確認。質量換算は約6.12kg。荷重位置と方向を記録。
5. 一次合格目安：亀裂・脱落・締結の緩みなし、無負荷へ戻した後の残留変位0.1mm以下、引っ掛かり増加なし。0.3mmの設計上の軸方向遊びと弾性変形は区別して測る。

この60Nは2kg×約3gを丸めた設計荷重であり、試験未実施の現状では許容荷重表示に使えない。

## 3. 上部架台と駆動

- 架台を組み、軸受の蓋・M3クロスピン・端部座金の保持を確認する。
- 2kgのダミー荷重を重心偏心50mm以内で固定し、まず手動で全範囲を動かす。電源を入れる前に全姿勢でケーブルとベルトの逃げを測る。
- ベルトを掛ける前後で抵抗を比較し、回転が重くなる場合は張力・軸の平行度・軸受保持を調整する。メーカーが指定するベルト張力を確認して設定する。
- 低速の左右±30°、前傾0～12°、傾き±6°を単軸で10往復し、続いて組合せ100往復で異音、歯飛び、摩耗粉、ねじ緩み、温度を記録する。
- 100往復は初期不具合確認であり、耐久寿命の保証にはならない。温間でクラッチの滑りトルクを再確認する。

## 記録

写真、材料、造形方向／加工仕様、重量、荷重位置、保持時間、測定器、温度、変位、滑りトルクを残す。未測定値は空欄にし、推測でPASSを付けない。
''')
rows=[['test_id','status','specimen','load_N','lever_mm','measured_displacement_mm','residual_mm','slip_force_N','temperature_C','notes'],['CLUTCH_CW','NOT_RUN','assembled_clutch','','50','','','','','5 measurements required'],['CLUTCH_CCW','NOT_RUN','assembled_clutch','','50','','','','','5 measurements required'],['AXIAL_DOWN','NOT_RUN','yaw_retention','60','0','','','','','step20/40/60N'],['AXIAL_UP','NOT_RUN','yaw_retention','60','0','','','','','reverse fixture'],['COMBINED_LOAD','NOT_RUN','yaw_retention','60','50','','','','','3Nm'],['UPPER_FRAME','NOT_RUN','upper_neck_frame','19.6133','50','','','','','2kg dummy,CGmeasured'],['INITIAL_CYCLE','NOT_RUN','complete_neck','','','','','','','100cycles not lifetime proof']]
with (r/'test-results.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
(r/'AUDIT_CORRECTION.md').write_text('''# モーター間干渉判定の訂正

v4～v6の一部探索は、ローカル中心が(12.5,-19,0)のOBBに `applyMatrix4` を使っていた。使用中のThree.js実装は、この呼び出しで中心へ平行移動を加えるため、原点から離れた中心を回転した正しい位置になっていなかった。これは本プロジェクトの変換の使い方の誤り。

これらの古い「モーター同士の干渉なし」という結果は取り下げる。プロジェクタ頂点を直接変換した収納検査まで同じ原因で無効になるわけではないが、新配置については新しい検査結果を使う。

v7は中心を点として剛体変換し、回転行列を別途設定する。90°回転の既知解・8頂点・交差の基本テストを追加した。採用配置はさらに、実際に回転したボックスの三角形BVHによる表面交差検査と照合する。マウント金具、配線、連続姿勢、実物公差まで合格したという意味ではない。

コード：`work/buildable-bust-v7/rigid-obb.mjs` と `rigid-obb.test.mjs`。最終結果はv7のJSONを見る。
''')
print('spring specification and NOT_RUN physical test records written')
