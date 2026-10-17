現在の候補は [成人サイズ・実機一式の配置](ADULT_FULL_PROJECTOR_DESIGN.md) を参照。頭部110%、底面実測135 × 75 mm。本体一式で検証しています。以下には過去段階の記録を含みます。

# TripoAIプロジェクター全体モデル

Downloadsの `projector module 3d model.glb` を読み込み、TI DLPDLCR3010EVM-G2の組立図revision Cの外接寸法 **135 ×81.34 ×49.82 mm** に合わせました。元ファイルは変更していません。

- 補正モデル：`assets/tripo-projector/projector-calibrated.blend`
- 補正GLB：`assets/tripo-projector/projector-calibrated.glb`（glTF標準のm単位。長辺0.135 m）
- 補正係数・軸変換：`assets/tripo-projector/calibration.json`
- 表示・遮蔽計算は963,236頂点、1,927,085三角形の全体モデルを使用。
- ユーザーの灰色の固定具は外す前提。AIモデルに元から含まれるメーカー側の保持部品は残す。

AI再構成は機械図面ではなく、外接寸法を合わせても各部品・レンズの位置が正確になるとは限りません。レンズ原点は公式CADから対応付けた機械的前縁中心であり、実測校正が必要です。公式STEP revision Dの立ち上がったフレックスを含む高さは約57.9 mmです。収納探索ではこの公式CADも併用しています。

現行の表示は**全体モデルを使った再設計途中の配置**です。従来の光学エンジン単体の100%照射結果は、本体一式の達成結果として扱いません。外形保持・本体収納・全顔照射が同時に成立する配置は探索中です。

[TI公式CAD配布元](https://www.ti.com/tool/TIDA-01571)

## 実測寸法（2026-09-21）

ユーザーのノギス測定：底面の長方形は **135 × 75 mm**。レンズ等の突出部は含まない。公式STEPの `3010-EVM-G2-ENGINE-BASE-SS` は135 × 75 mm（厚さ3.97 mm）で一致する。全体幅約81.34 mmと底面幅75 mmは異なる測定対象のため、プロジェクタ全体を75 mm幅に縮小しない。TripoAI形状の局所寸法やレンズ光学基準点は未実測。
