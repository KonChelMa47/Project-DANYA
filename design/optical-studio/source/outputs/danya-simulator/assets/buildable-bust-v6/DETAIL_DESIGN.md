# DANYA 左右軸 v6 — 接続・抜け止めの試作詳細

2026-09-22。対象は左右回転の歯車ユニット。v5の仮穴・6mm歯幅・旧支持軌道を置き換える。**現物強度を保証した製作承認図ではない。上部キャリアとの接続、サーボ本体の固定フレーム、全体の配線は未完成。**

## 1. 確定したCAD寸法

単位mm。STEP/STLはXYが各部品の原点、Zは組立座標。`parts.json`のxyを加えて配置する。STLに単位情報はないためmm指定。

|部位|寸法／指定|
|---|---|
|大歯車／小歯車|96T／20T、m1、20°、減速4.8、軸間58|
|歯幅|各10、軸方向重なり9.5|
|大歯車|外径98、内径76、Z −24.5～−14.5|
|小歯車|外径22、滑り穴8.2、Z −25～−15|
|ホーンアダプター|フランジ径26×5、軸径8、4穴PCD14、通し3.3、皿径6.7|
|独立保持ボルト|M3×30六角、頭をAF5.7ポケット内に保持。サーボの中央ねじとは別系統|
|固定スリーブ|POM、外径75.6／内径72。直径隙間0.4|
|下側支持板|6061-T6、厚3、Z −28～−25。6つの取付耳、PCD108、穴3.3|
|上側保持板|6061-T6、厚3、内径92／外径116、Z −13.7～−10.7|
|固定板間スペーサー|金属、外径8／穴3.3、長さ11.3、6本|
|スラスト材|下PTFE0.5、上PTFE0.5。上方向遊び0.3|
|出力取付|半径42の2穴、3.3通し。現在の6mmスペーサーはキャリア側接続の境界部品|

大歯車の外周は歯先なので、上のPTFEは歯先側の捕捉面として使う。通常の全周スラスト軸受と同じ負荷分布とはみなさない。偏荷重の実試験が必要。

## 2. サーボとの接続

WaveshareのメーカーSTEPの金属ホーン（solid7）から、径19.2・PCD14・4穴径2.5を抽出した。**STEPの径2.5だけではM3ねじとは断定できない。** 受入時にM3×0.5ゲージで確認し、未加工ならホーン単体を外して加工・洗浄する。異なるねじ規格には無理にねじ込まない。

付属ホーンと純正中央ねじを使う。印刷スプラインは作らない。独立M3保持ボルトをアダプターの裏側から入れ、その後ホーンにアダプターを4本のM3×8皿ねじで固定する。アダプター厚5に対し突出3、CAD上の穴周辺肉厚約3.56なので約0.56の余裕。実物でねじ先がサーボ本体に触れないことを確認する。ねじを長くして代用しない。

## 3. 小歯車の抜け止めと過負荷保護

下から、アダプター → 摩擦材0.4 → 小歯車10 → 摩擦材0.4 → 鋼ワッシャ0.8 → 圧縮ばね → 上ワッシャ0.8 → M3ナイロンロックナット。保持はM3×30ボルトが行う。接着剤・圧入だけで保持しない。

ばねは外径8／内径6／自由長16を選定寸法とする。線径1、有効巻数4程度なら概算ばね定数約5.8N/mmだが、**購入品のばね定数・密着長を確認し、回転トルクで調整する**。ガイド外径5.6、長さ6.5。組付けばね長8を出発点とし、密着・ガイド底付きまで最低1mm残す。摩擦係数や締め付け回数からトルクを保証しない。適合するばねと摩擦材の品番は未確定。

クラッチ滑り始めは**入力0.08±0.01N·m**。0.10N·mを超える設定は禁止という試作条件。半径50mmの治具なら1.4～1.8Nで動き始めるよう調整する。時計回り・反時計回りを各5回、慣らし後と温間で再確認。滑った後はサーボ内の角度と頭の角度が一致しないため、必ず原点合わせをやり直す。自動復帰制御は未実装。

## 4. 大歯車の抜け止めとねじ長さ

下支持板3＋スペーサー11.3＋上保持板3＝17.3。M3×25皿ボルトを下から6本使用。下板の皿穴（径6.7／深さ1.7）で下面を平らにする。上側に0.5ワッシャ、ナイロンロックナット厚4として総スタック21.8、ねじ先余長3.2。位置角は0°から60°刻み。実際の購入ワッシャ・ナット厚で再確認する。

板間距離は金属スペーサーで決め、樹脂歯車を締め付けない。歯車の上方向遊び0.3、径方向片側0.2を確保。動作が重い場合は歯を削って逃がす前に板の平行度とスペーサーを測る。取付面の平行度目標0.1、スペーサー長公差±0.05、軸間距離目標58±0.1。印刷穴・軸は加工後の寸法で合否を判定する。

上部キャリアの既存柱は上保持板と干渉するため画面から除去した。半径42の出力穴から新キャリアへ荷重を渡す設計は継続項目。2本の仮スペーサーだけに頭を載せて使用できる状態ではない。

## 5. 強度計算の範囲

想定条件は可動部2kg以下、重心偏心50mm以下、荷重倍率3。軸力60N・転倒モーメント3N·mを試作設計荷重とする。これらは実測値ではない。通常の左右駆動トルク目標は出力0.20N·m以下。重量を支える軸受摩擦だけでこれを超える場合、クラッチを強く締めず軸受方式を見直す。

- 歯面接線力：入力上限0.10N·m／半径10mm＝10N。
- 簡易歯元曲げ：2×10／(歯幅重なり9.5×m1×仮定歯形係数0.30)＝7.02MPa。
- 仮のPETG許容8MPaに対する倍率1.14。**この許容値はメーカー保証値ではなく設計仮定で、余裕は小さい。** 積層方向・温度・疲労・クリープ・歯元の造形欠陥は未評価。現在の根元は簡易形状で、量産ホブ歯形のフィレットではない。
- クラッチなしでサーボ最大約2.94N·mを受けると同じ簡易式で約206MPaとなり、この樹脂案は不成立。
- 保持板の局部荷重100N、幅12・片持ち長8・厚3の近似では44.4MPa。仮定許容80MPaに対する倍率1.8。ただし切欠き・環状板全体の変形はこの式で証明できない。
- M3の引張有効面積5.03mm²に100Nを仮定すると19.9MPa。ボルトより樹脂や取付部の変形が支配し得る。ハブの薄肉部・出力穴の引き抜きは実試験対象。

計算値は`strength-calculation.json`。これはFEA、ISO/AGMAの正式強度評価や疲労寿命計算ではない。

## 6. 試作の合否確認

1. 頭・プロジェクタを付けず、手回しで左右±30°、歯先接触・片当たり・ねじ先接触がないこと。
2. クラッチの滑りトルクを校正。温間で上限を超えないこと。
3. 別治具に保持ユニットを取り付け、設計軸力60Nと転倒モーメント3N·mを段階的に負荷し、割れ・脱落・ねじ緩み・残留変形がないことを測定。頭を載せて人の前で試験しない。
4. 実際の可動部の重量・重心・駆動抵抗を測る。仮定超過なら再計算。
5. 耐久運転後に歯面、保持面、ハブの座面、ナットの緩みを再確認。耐久回数・寿命要件は未確定。

## 7. 予算と製作状態

部品は発注していない。モーター11,000／制御1,500／鏡5,000／電源3,000／軸受・歯車・保持板・クラッチ5,000／配線1,500／ねじ1,500／送料等1,500＝30,000円の上限割当。加工板・ばね・鏡の見積未取得のため、購入総額を保証しない。印刷材料と既存プロジェクタは枠外。

## 出典

- [Waveshare ST3215公式仕様・ねじ先接触の注意](https://www.waveshare.com/wiki/ST3215_Servo)
- [メーカーSTEP](https://files.waveshare.com/upload/5/59/ST3215-3D.zip)。穴抽出記録は作業ディレクトリのhorn-audit.json。
- [KHK 平歯車強度の技術資料](https://khkgears.net/pdf/spur-tech.pdf)。本書は正式規格計算ではなく簡易スクリーニング。

## CAD一覧

- [horn-clutch-hub.step](horn-clutch-hub.step) · [STL](horn-clutch-hub.stl) — PETG prototype
- [pinion-20T-clutch.step](pinion-20T-clutch.step) · [STL](pinion-20T-clutch.stl) — PETG prototype
- [yaw-ring-96T-retained.step](yaw-ring-96T-retained.step) · [STL](yaw-ring-96T-retained.stl) — PETG prototype
- [yaw-bearing-base.step](yaw-bearing-base.step) · [STL](yaw-bearing-base.stl) — 6061-T6 aluminium 3mm
- [yaw-radial-sleeve.step](yaw-radial-sleeve.step) · [STL](yaw-radial-sleeve.stl) — POM or machined low-friction polymer
- [lower-thrust-liner.step](lower-thrust-liner.step) · [STL](lower-thrust-liner.stl) — PTFE sheet
- [upper-retaining-plate.step](upper-retaining-plate.step) · [STL](upper-retaining-plate.stl) — 6061-T6 aluminium 3mm
- [upper-thrust-liner.step](upper-thrust-liner.step) · [STL](upper-thrust-liner.stl) — PTFE sheet
- [clutch-lower-friction.step](clutch-lower-friction.step) · [STL](clutch-lower-friction.stl) — friction sheet
- [clutch-upper-friction.step](clutch-upper-friction.step) · [STL](clutch-upper-friction.stl) — friction sheet
- [clutch-pressure-washer.step](clutch-pressure-washer.step) · [STL](clutch-pressure-washer.stl) — steel shim
- [spring-guide.step](spring-guide.step) · [STL](spring-guide.stl) — PETG prototype
- [retainer-spacer-1.step](retainer-spacer-1.step) · [STL](retainer-spacer-1.stl) — steel or aluminium
- [retainer-spacer-2.step](retainer-spacer-2.step) · [STL](retainer-spacer-2.stl) — steel or aluminium
- [retainer-spacer-3.step](retainer-spacer-3.step) · [STL](retainer-spacer-3.stl) — steel or aluminium
- [retainer-spacer-4.step](retainer-spacer-4.step) · [STL](retainer-spacer-4.stl) — steel or aluminium
- [retainer-spacer-5.step](retainer-spacer-5.step) · [STL](retainer-spacer-5.stl) — steel or aluminium
- [retainer-spacer-6.step](retainer-spacer-6.step) · [STL](retainer-spacer-6.stl) — steel or aluminium
- [carrier-standoff-1.step](carrier-standoff-1.step) · [STL](carrier-standoff-1.stl) — steel or aluminium
- [carrier-standoff-2.step](carrier-standoff-2.step) · [STL](carrier-standoff-2.stl) — steel or aluminium

## 保存した検査結果

- [packaging-check.json](packaging-check.json)：20部品とプロジェクタ：27姿勢の表面交差なし。
- [motor-check.json](motor-check.json)：下げた左右サーボ：27姿勢のモーター包絡検査で交差なし。
- [mesh-check.json](mesh-check.json)：181段階の歯面交差なし。
- [pair-check.json](pair-check.json)：選択した歯車/ハブと固定部の3左右角で交差なし。
- [fastener-check.json](fastener-check.json)：購入ねじ等7組の円筒包絡とプロジェクタ：27姿勢で交差なし。
- [light-check.json](light-check.json)：間引いた主光線と新部品：27姿勢で遮断なし。有限光束の検証ではない。

[組立配置STEP](yaw-unit-assembly.step)（20部品。購入ねじ・ばねは別途部品表） · [購入/加工部品表](HARDWARE.csv)
