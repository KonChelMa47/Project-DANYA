# 胴体内プロジェクタと3軸可動首：検討中

2026-09-22。ユーザー指定により頭部内収納限定を撤回し、胴体内収納・追加凸レンズなし・首の可動を条件とする。頭部と顔の外形は維持。

試作画面：/neck-study.html。左右±45°、うなずき±20°、傾き±15°を操作できる。これは提案する評価範囲であり、実機の保証可動範囲ではない。

頭、平面鏡2枚、実寸プロジェクタ全体を同じ剛体フレームで動かす構成。相対位置が変わらないため理想光学系の投影関係は維持される。画面の支点・座標軸は運動学の説明であり、完成した機構を意味しない。荷重、重心、駆動トルク、軸受、配線、放熱、フレームの変形は未設計。

TI一式とTripo全体の包絡直方体を8177姿勢で回転。可動範囲の軸平行包絡は奥行299.38×幅307.24×高さ165.67 mm。各面10 mmの余裕を足した参考箱は奥行319.38×幅327.24×高さ185.67 mm。実形状の連続衝突検証ではなく、首・頭部・ミラー・支持材・光束を含む完成胴体の外寸でもない。非対称の包絡箱であり、首中心に対称な胴体ならさらに大きくなる。

新しい平面鏡用の内壁で、顔の頂点＋面重心255542点中254817点が壁厚と公式本体を含む代表光線検証を通過（99.7163%）。725点が未達。全顔100%達成の主張はしない。光路長327.04–450.96 mm。

仮定した射出瞳径2–8 mm、合焦距離350 mmの有限光束検証は103視野点のみ。主な残課題は遮蔽と口径食。未マスク全画面の迷光検証もaudit.jsonに保存してあり、現在の構成に旧凸面鏡用のマスクを流用しない。実際のオーキャン映像の投影ワープは既存シミュレーターに残っているが、この運動学専用画面には未統合。

従来のメイン画面は旧凸面鏡案。新構成へのリンクを追加したが、旧案を新案として表示していない。原本STLは変更していない。

次の作業：725遮蔽点の特定と内壁修正、有限光束と二重像評価、胴体の小型化、首の支持機構と連続干渉評価、実映像ワープ統合。


## 内面修正と迷光検証の更新

新しい候補を首の試作画面へ反映。顔の外側169900三角形は元データと完全一致。内面の平滑化と局所的な縁の修正で、予定光路255542/255542点が通過した。最大内面移動6.5826 mm、2.6 mm以上動かした頂点は92。開いた辺・非多様体辺は0、Blender EXACTの頭部との材料交差は0。ただし自己交差と製造可能な最小肉厚は未確定。

別案の放射方向の内面再構築も100%を通過したが最大約40mm動くため採用していない。現在の候補は元の内面に対する平滑化案。

非逐次の最近接面判定では1023本中1018本が顔、5本が頭と判定。5本の到達点は顔・後頭部の共有境界そのものと5e-13 mm以内で一致し、面の選択が競合している。これを内部遮光とは同一視しないが、全連続面の照射証明でもない。

射出瞳径2/4/6/8mm・合焦距離350mmを仮定し、各1023視野×64光線を評価。8mmの場合、顔へ58796/65472、頭412、開口外2923、鏡背面3341。光束の欠けが残る。実機の射出瞳と合焦範囲を確定する測定ではない。

放射内面案を用いた1280×720マスクは各画素9点の主光線判定で作成し、全顔サンプルの消失は0。これを平滑化内面へ適用した有限光束の別検証では、8mm/350mm仮定で134448光線を評価し別経路の顔到達0を検出した。ただし疎なサンプルであり、二重像不発生の保証ではなく、拡散・多重散乱・通常の裏面鏡のゴーストも含まない。マスクの試作画面への映像統合は未完了。

## 胴体小型化 v3（現在の表示）

ユーザーの指定により、目・鼻・口の映像位置合わせを後回しにし、胴体の収納を優先した。

首の可動範囲を左右±45°、上下±20°、傾き±15°のまま維持。支点を[5,0,10]mm（頭部下端から32.1mm下）へ移し、全体一式のプロジェクタを[-23.1078,-17.8731,-56.5250]mm、Euler XYZ[-63.8836,-35.6268,-146.0937]度へ変更。小鏡は23.8282×35.1849mmの平面矩形。大鏡と顔・後頭部の外形は維持。

両方の完全な本体メッシュの全1627768頂点から309頂点の凸包を求め、小鏡の厚さ・四隅・レンズ原点も加えた。8177姿勢を2.5度刻みで評価し、角度間の連続変化には座標あたり0.3498mm以下の補正を適用。各面10mmの余裕を加えた収納見積もりは幅253.8503×奥行255.1224×高さ164.0613mm、10.625L。元の配置も同じ方法では17.176Lで、体積38.1%減。元の単純な本体直方体の見積もりとの差を改善量として混ぜていない。

補正根拠：連続回転角の座標極値では内点の一階偏微分は0、境界角は格子上にある。回転の各二階偏微分の絶対値は支点からの最大半径R以下なので、格子の半刻みδに対する誤差はR(3δ)^2/2以下。評価範囲は上記Euler ZYXの直方体であり、実機のバックラッシュやたわみを意味しない。

収納見積もりはモーター、軸受、支持材、ケーブル、放熱部品、外装厚さを含まない。完成外装・製作用CADではない。頭部外面の連続全可動域での最低高さはz=5.9400mm、収納箱上面z=3.8016mmとの差は2.1384mm。これは外形同士の理想幾何での分離であり、首機構や実機公差を含む検証ではない。

顔の予定光路は255542/255542点が通過。両本体で8mm仮定瞳・350mm仮定合焦の65472光線中64831が顔、471が頭、170が開口外。旧試験マスクをこの配置で134448光線検証し、別経路の顔到達は検出0。167698頂点で旧映像UVとの差は最大2.981e-8なので表示用の全顔変換データを再利用した。有限光束・実機合焦・透過率は保証していない。

プロジェクタ固定＋単純な可動平面鏡で頭の3軸動作を補償する案は、探索した条件では最悪姿勢の照射範囲が不足したため採用していない。1軸だけ本体を回す派生案も同様。これらを全方式の不可能性の証明とはしていない。

次はこの小型配置を基準に、首元の光路開口、首機構・支持材・配線と放熱を含む実際の外装を詰める。顔のランドマーク位置合わせは後回し。


## 胴体外装と首開口の追加検証

v3を維持し、固定外装の試作STLを出力した。外寸は幅253.850×奥行255.122×高さ164.061mm、壁・底・天板3mm、天板の首開口は前後101×左右145mm。開口は全可動域の両平面鏡間の光束包絡から計算し、角度補間の座標補正1.27781mmと開口の追加余裕4mmを含む。外装メッシュは24頂点44三角形、開いた辺・非多様体辺0、体積は解析式と一致。

出力メッシュを独立に検証：105姿勢、顔343点、光路108045区間で外装による遮断0。27姿勢×2種類の全体プロジェクタモデルの計54干渉確認で交差0。連続包絡の検証を補う疎な独立検査であり、実機の公差や首支持機構を保証しない。

首支点を前方へ移す8候補も比較した。最小の疎な格子評価では9.102Lだが、頭の最低位置が水平な外装上面を下回る。これだけで実際の交差を断定しないが、現行の上下面分離条件を満たさないため未採用。現行支点の鏡間光束包絡からの距離は約9.01mmで、中央に大きな首機構を置けるとは限らない。前方移動はこの空間を広げる一方、上部外装の形状と頭の動作の再検証が必要。

試作STL：assets/torso-flat-study/DANYA-torso-envelope-prototype.stl。首開口を覆うカバー、首機構、支持材、配線、放熱、締結は未設計であり、製造用の完成形ではない。目・鼻・口の位置合わせはユーザー指定どおり後回し。


## 継続探索：v4候補（未公開）
前ターンは外装検査と支点比較の証拠を追加したprogress。今回は支点[5,0,0]候補を連続補正付きで評価。可動範囲は維持し、外寸候補は幅245.5599×奥行247.0592×高さ163.7139mm、9.93217L（v3より6.52%減）。頭部内外壁頂点の最低高さ5.01676mmに対し外装上面2.86030mm、上下分離2.15647mm。鏡間光路を585姿勢・384セルの補間補正1.32965mmで包絡し、3mm天板の開口を103×151mmへ拡大。候補STLはwork/torso-flat/v4-housing/。開いた辺・非多様体辺0、解析体積一致。公開版は検査済みv3のまま。
首の軸配置を光学的に予備検討：支点から130mmの直線で、左右・前・下は鏡間凸包との連続距離下限10mm超、後方・上方は凸包へ侵入。これは本体やブラケット、複数軸の相対回転を未評価の候補選別で、首機構成立の証明ではない。v4-shaft-optical-screen.json参照。
v4の独立外装BVH検査はexec session62040で実行中。結果確認後に公開版変更を判断する。首支持・駆動・配線・放熱・実光学の不確定性が残り、goalはactive。


### v4首機構の予備配置（継続）
左右85mmのピッチ軸中心線を、内側ロール軸の±15度について0.5度刻み、中心線0.5mm刻みで検査。座標距離のLipschitz補正0.62099mmを差し引いても、鏡間凸包から左右それぞれ9.4465/10.2179mm、全体本体凸包から18.4739/39.4866mm以上離れる。径8mmの仮軸はこれら2種類の包絡には当たらない見込み。ただし鏡実体、頭壁、軸受、支持アーム、荷重は別途未検証。軸方向Xの後方は光路、Z下方は本体へ入るため、単純な中央下向き軸は不採用。前方片持ちロール＋左右ピッチ＋外周ヨー支持を次に検討する。
固定屋根の現在開口は光路用のみ。例えば左右55mmの軸でもヨー45度でx約44mmまで回り、開口のx最大21mmを越える。このため径8mm軸を含む首開口の再設計が必要で、機械軸を現在の公開モデルに無条件で追加していない。
検証ソース: v4-pitch-axis-optical.mjs/json, v4-pitch-axis-body.mjs/json, v4-shaft-body-screen.mjs/json。v4外装audit session62040はこのターンにpoll後もlive、ps PID14015 CPU93.9%経過1:39で確認。再起動せず次回同handleを確認する。公開版は依然v3。


### v4左右軸用の開口拡張
前ターンは首軸の光束・本体との距離評価を実行したprogress。今回は直径8mm長さ170mmのピッチ軸を左右に設ける仮定で首開口を155×184mmへ拡張（x[-82,73],y[-92,92]）。外寸はv4のまま。軸はRzRyのY軸でありpitchで方向が変わらず、rollは内側なのでshaftの高さ範囲はz±4mm。yaw±45度の連続包絡に3mm余裕を付けて開口を計算。work/torso-flat/v4-shaft-housing/に別候補STLを生成し、既存v4光路用外装は変更していない。
独立メッシュ検査181 yaw姿勢で軸と外装の交差0。光束凸包との補正済み距離から軸半径を引いて最低5.446mm、本体凸包から14.474mm、頭部最低zから1.0168mmの理想幾何間隔。軸受・駆動・腕・配線・荷重・実機誤差や他経路迷光は未検証。audit-pitch-shaft-housing.mjsとv4-shaft-housing-audit.json参照。公開モデルはv3のまま。
長時間v4本体外装検査session62040はpollでrunning、ps PID14015 CPU92.1%経過3:49を確認。次回同じhandleを継続確認。再起動していない。goal active。


## 現在公開：v4＋左右仮軸用開口
session62040は正常終了。housing-v4-integration-audit.jsonで105姿勢・108045区間の外装遮断0、54全体本体検査の交差0を確認。独立検査した光路用外装から屋根の材料だけを追加除去した155×184mm開口版を公開採用（新規の遮蔽物は増えない）。軸受等の成立は未証明。
公開assets/torso-flat-study/neck-envelope.jsonはpivot[5,0,0]、幅245.5599×奥行247.0592×高さ163.7139mm、9.93217L、元の配置比42.173%減。torsoバイナリとSTLをv4-shaft-housingから反映。torso-manifest.jsonに独立検査結果と材料除去による検証継承の範囲を添付。光学配置・顔映像UVは変更なし。v3以前の公開ファイルはwork/torso-flat/v3-published-backupに保存。
既存ユーザーtab9をreloadし、9.9L・42%減・支点下42mm、最新寸法と開口、オーキャンv2稼働をAXで確認。yaw45/pitch20/roll15でスクリーンショット確認後、正面に戻した。仮軸は表示していない（開口空間のみ）。検査プロセスは終了済みでpendingなし。
次の実作業：前方片持ちroll支持と左右pitch支持・外周yawを構造として接続し、軸受/モーター/支持材/配線/放熱/首カバーを含む収納検証。実機射出瞳・合焦・光束損失・迷光・薄肉シェル製造性は未解決。目鼻口の位置合わせはユーザー指定で後回し。goal未達成、active。前ターンも今回も証拠と成果物を追加したprogress。


### 首支持フレームの候補探索
前回はv4公開反映まで完了したprogress。今回はU字pitch carrierを3案（前辺x25/35/45mm）検査。候補x25: 中心線[5,-85,0]→[25,-85,0]→[25,85,0]→[5,85,0]、半径4mm。共通yaw/pitchを消去したhead座標でroll±15度、0.5度刻みと中心線1mm以下刻み、距離補正0.870983mmを適用。管外面までの連続距離下限は光束凸包24.860mm、本体凸包10.145mm、頭部壁26.019mm、顔壁39.996mm。さらに管全体のhead座標最大z25.9997mmは頭壁最小42.1055/顔壁最小47.3888mmより下で、材料内に含まれるケースを除外。yoke-screen.mjs/json参照。前案より検証対象を実際の接続部材へ拡張した。
U字carrierのyaw/pitch包絡＋管半径4＋余裕3mmを含む開口169×190mm候補を別途生成。v4-yoke-housing/のSTLは外寸を維持、open/nonmanifold0、解析体積一致。公開版はまだ左右仮軸用155×184開口。次はyokeと外装の独立干渉検査と、前側roll軸受および外側yaw/pitch支持の接続。構造荷重、鏡/プロジェクタの取付、モーター、ケーブル、冷却、首カバー、実機光学は未確定。進行中プロセスなし。goal active。


### 公開更新：U字支持候補と169×190mm開口
前ターンはU字支持材の連続距離と開口包絡を追加したprogress。今回はaudit-yoke-housing.mjsを実行、yaw37×pitch17の629姿勢、4円柱の計2516組で外装交差0。結果yoke-housing-audit.json。仮軸＋U字carrierの材質形状のみであり、軸受/締結/荷重/駆動やhead/projectorへの接続は未成立。
v4-yoke-housingの外装バイナリ/STLを公開へ反映。外寸9.93217Lは据置、開口169×190mm。既検査外装からroofを追加除去するだけである。torso-manifest.jsonへ支持材の距離・独立外装検査を統合。
neck-studyに任意の「首支持フレームの候補を表示」を追加。carrierはRz(yaw)Ry(pitch)で回り、内側optical rigはRzRyRx(roll)。配置検討用、軸受・駆動・固定部未設計を明示。既存tab9reload、checkbox選択、45/20/15姿勢のスクリーンショットを確認後正面へ戻した（候補表示on）。映像稼働確認。処理pendingなし。次はfront roll軸受と光学rigへの接続、外側pitch/yaw支え、荷重・駆動・配線・放熱・首覆いを含む成立性。goal active。


### 軸受ケースの候補評価
SKF公式608-2RSHの8×22×7mmを調査（https://www.emarketplace.in.skf.com/deep-groove-ball-bearing/608-2rsh）。購入/設置なし。外径28mmケースを仮定し、rollケースx12..38,y=z0、左右pitchケースx5,y75..95/−95..−75,z0の保守的カプセル包絡を評価。roll±15度、軸0.5mm刻み、距離補正0.664616mm。最小光束間隔roll2.945mm、最小本体間隔右pitch4.731mm。その他head/faceも正。全カプセル最高z38.588mm<頭壁最低42.106mmで材料外を確認。bearing-sleeve-screen.mjs/json。
現行左右シャフト±85mmは中心y±85の608幅7mmを最後まで貫通しないため延長必須。現行U字前辺はroll軸受部分で中断し、穴付きケースへ置換が必要。ケース包絡の成立を機構完成と誤認しない。公開モデルは軸受なしのU字候補のまま。
重量をasync質問（プロジェクタ・印刷顔/頭の各質量、不明回答可）、まだ未回答。寸法作業は継続可能。pendingプロセスなし。前回も今回もprogress。次はシャフト延長と軸受/留め具を含む接続・外装干渉。goal active。


### 公開更新：190mm左右軸＋軸受ケース必要空間
前回は軸受包絡を検証したprogress。今回は軸を±95mmへ延長、相対rollの連続補正後の中心線距離は光束最小9.40284、本体18.4303mm。径8mmを引いても正。軸受中心±85、幅7の両端±88.5を貫通するが抜け止め・段付き軸は未設計。
169×200開口候補を生成。左右ケース外包円柱半径14、軸方向y75..95/−95..−75。yawで|Y|<=sqrt(95²+14²)、X<=5+(95+14)/sqrt(2)、余裕3mmを加えて屋根を拡張。rollケースx12..38、半径14も含めた独立検査audit-bearing-housing.mjsで629姿勢4403組交差0。ケースは保守的な中実包絡、実際の穴・締結は未設計。body/head距離は前回検証済み。
公開外装バイナリ/STLをv4-bearing-housingへ更新、外寸は維持。画面の左右軸を190mmに延長、前辺中心をケースのため中断し、ケース必要空間を灰色半透明で表示。これを完成機構と表示していない。torso-manifest.jsonへ検査統合。既存tab9でreload、支持表示on、スクリーンショットでライブ顔映像・ケース表示確認。重量質問未回答、処理pendingなし。次は実際の穴付きケース/締結と外側支持接続、光学rig取付、駆動・強度・配線・冷却・首覆い。goal active。


### 外側yaw支持の探索
前回は軸延長/軸受必要空間の公開反映まで完了したprogress。今回、左右pitch軸受から下へ伸ばす外側yawフレームを探索。x5、y±100/110の垂直柱は本体凸包と交差し不採用。x5,y±115はサンプル距離4.743mmのみで連続補正付き成立を確認できず不採用。柱をx−35,y±110へ移した後方オフセット案が有望。
1度pitch×1度roll、中心線2mm以下間隔で再評価。前記柱+上下接続5線、半径4mm管、位置補正4.18503mmを差し引いても本体凸包から3.48479mm、光束凸包から51.1735mm以上離れる。outer-yaw-support-refined.json参照。yawは共通運動なので相対距離に影響しない。head/face壁の距離評価は別プロセスsession18249で実行開始、次回同handleをpoll（未完了なら再起動しない）。
yaw±45の柱包絡と3mm余裕+3mm壁を収める外装見積もりは幅254.094×奥行256.592×高さ163.714mm。outer-yaw-support-case-estimate.json。現在公開245.560×247.059×163.714mmに対し、支持柱まで含めると寸法が増える。未検証候補のため公開モデルはまだ変更なし。支持脚・yaw軸受/駆動・締結・光学rig取付・配線/冷却/首覆いは未確定。重量質問未回答。goal active。


### v5支持柱込み外装候補（未公開）
session18249正常終了。後方x−35,y±110,z0..−140のyaw支持管と接続部はhead/face表面から補正・管半径を引いて最低24.4208/54.0294mm。相対pitch/rollの連結範囲で表面を横断せず、初期全管z<=4<全head/face最小zなので材料内部に含まれる可能性も除外。
柱の連続yaw包絡に光路・軸受開口を合成し、3mm隙間・3mm壁・最小2mmの非構造屋根縁を追加したv5試作外寸は幅260×奥行259.5258×高さ163.7139mm。屋根開口196×250mm。前ターンの25.4×25.7cm見積もりから縁と整数mm開口の余白分増加。ケース形状・首覆いはさらに検討が必要。
v5-post-housing/にSTL・bin・QA。open/nonmanifold0、解析体積一致。独立検査v5-inner-housing-audit.jsonは内側支持+軸受包絡629姿勢4403組交差0、v5-outer-housing-audit.jsonは外柱181姿勢905組交差0。body/光束への距離はouter-yaw-support-refined、head/faceはouter-yaw-support-shells。既存光学収納領域を縮めず拡張したケース。
未実施：inner/outer支持部材同士の干渉（意図した接合/軸受接触と区別）、yaw土台接続、軸受締結、光学rig支持、荷重/駆動/配線/冷却/首覆い。公開はv4+軸受必要空間のまま。全process終了pendingなし。重量未回答。前回/今回ともprogress。goal active。


## 現在公開：v5支持柱込み外装＋内側U字修正
接続確認で旧U字端y±85がpitchケースy75..95の側壁を横切る問題を発見。U字をy±70へ変更し、半径4込み|y|<=74、ケース内端75との理想軸方向隙間1mmを確保。yaw/pitchで相対yは変わらない。yoke-inboard-screenで光束/本体/head/faceとの連続距離下限24.9257/10.3308/22.7739/40.0617mm。旧case支持アームは中心から伸びていたが、元の同一線分上x−9,y±93.75で切り詰め、ケース外面接続候補へ変更。切り詰めは検証済み管の部分集合。
mechanism-self-clearance.mjsでinner支持/shaft/rollケース vs outer5本管の線分間最短距離を正確に計算、pitch0.5deg刻みと連続補正。最小下限はshaft5.5855mm、U腕19.2516mm、front32.7468mm、rollcase77.8677mm。軸受とshaftの意図した接触・固定部の重なりは別扱いで、締結や軸受穴/肩/抜け止め未設計。
公開外装をv5-post-housingへ変更。寸法260(Y)×259.5258(X)×163.7139(Z)mm、開口196×250。manifestへ各検証追加、mechanism-outer.jsonを追加。表示の枠/外装寸法はcaseSpecに基づき、本体光学収納9.9Lとは明確に分けた。outerCarrierはyawのみ、innerCarrierはyaw+pitch、opticsはyaw+pitch+roll。既存tab9で最大45/20/15のスクショ確認後正面へ戻し支持表示on。ライブ映像あり。
全job終了pendingなし。今回も前回もprogress。次は底部yaw支持/旋回軸と光学rigをroll軸へつなぐ支持、実軸受ケース/締結・荷重/駆動・配線/冷却・首覆い。全顔照射の実機射出瞳/合焦/透過/迷光不確定性も残る。重量質問未回答。goal active。


### 底部yaw接続の探索
前回はv5公開と機構同士の隙間修正を実施したprogress。今回は後方の下桟x−35,y0,z−140から中央軸x5,y0へ回り込む3線分（下降、水平、軸）の候補を検査。baseZ−155/−160/−165、直径8mmで相対pitch/roll1度、中心線2mm以下、連続補正を評価。最小base−155で本体との管外面距離下限16.638mm、光束62.232mm。yaw-base-link-screen.json。部材長さ26mmの下向き軸までを含む。
さらに浅いbaseZ−140/−145をsession62464で検査中。次回同handleをpoll。支持経路の成立と、底部軸受ケース（仮外径28mm・長さ22mm、上下に608相当を配置）の収納/荷重成立は別。水平材の下面と固定ケース上面の擦れを避ける隙間も必要で、ケース上面を単純にbase−4にする案は採用しない。軸受・抜け止め・土台フランジを含む高さは未確定。
公開は変更なし、v5外寸260×259.526×163.714。重量質問未回答。goal active。

追記：62464は終了。base−145は管外面の連続距離下限8.52084mm（本体）を得た。base−140では下降線分が長さ0となるサンプリングの0除算を発見したため、その初回結果は無効。n>=1へ修正しsession63734で再実行中（これが次回poll対象）。NaN由来の初回ゼロ距離を物理的干渉と解釈しない。


### yaw軸受ポケット試作と有限ケース検証
63734正常終了。ゼロ長さ線分の0除算修正後、baseZ−140でも管外面の本体距離下限5.06905mm、光束59.08025mmで通過。−145は8.52084mm。初回NaNに基づくゼロ距離は無効で置換済み。両案とも棒材の経路のみでケースの太さは別。
yaw-case-body-screen.mjsをsession40559で実行中。base−140/−145、軸受ケースを半径14高さ22の有限円柱、上面base−6/下面base−28としてbody凸包との最近接三角形距離を1271 pitch/roll姿勢で評価。最近格子回転と円柱多角形誤差の補正を含む。次回は40559をpoll、再起動しない。
work/torso-flat/yaw-bearing-pocket/bearing-pocket-dimension-prototype.stlを新規生成。外径28高さ22、両端22.2径×7.2深さ、中央貫通8.6径。608規格の寸法を基準としたクリアランス仮定の試作。768頂点1536三角形、非多様体辺0、解析多角形体積と一致。実際のはめあい・抜け止め・分割締結・負荷/印刷補正未確定、完成軸受ブロックではない。build-bearing-pocket-prototype.mjs、同manifest.json参照。
公開v5は変更なし。前回/今回ともprogress。重量質問未回答。goal active。


### v6底部ケース収納候補（未公開）
40559は正常終了。有限円柱ケースr14,H22、base−140の中心[5,0,−157]でbody凸包とのサンプル最短9.99482mm、連続pitch/roll+64角形補正2.95918mmを引き7.03564mm。base−145は11.1340mm。yaw-case-body-screen.json。前ターンも今回もprogress。
予定光路の鏡前面2枚と仮8mm瞳正方形を含む包絡を連続姿勢で評価し、最低z−73.2183mm、ケース上面−146との垂直隙間72.7817mm。yaw-case-optical-height.json。別経路迷光/散乱・実瞳位置は含まない。
v6-yaw-base-housing/を新規出力。外寸幅260×奥行259.5258×高さ176.8603mm、床底−174/床内面−171、軸受ケース下面−168/上面−146、軸先端候補−170。既存棒材検証は−166までなので末端4mm追加部分は別途確認すること（body最低z−150.854より下なので垂直包絡で検証可能）。水平リンク下面−144、固定ケース上面−146の理想隙間2mm。ケースの土台締結/抜け止め/駆動は未確定。STLはopen/nonmanifold0、解析体積一致。
公開v5は未変更。次は末端・ケース/リンクの外装検証と土台への接続、反映判断。全process終了pendingなし、重量質問未回答。goal active。


## 現在公開：v6底部ケース込み
v6-base-housing-audit.jsonでyaw181姿勢×link/shaft/case=543組、外装との交差0。軸先端−170と床内面−171の隙間1mm、ケース下面−168と床内面の隙間3mm。追加軸末端−166..−170は全body最低z−150.8536より下で15.146mm以上のZ分離。case top−146とlink下面−144は2mmの理想隙間。固定・抜け止め・フランジや荷重は未検証。
公開外装をv6-yaw-base-housingへ更新、幅260×奥行259.5258×高さ176.8603mm。外側frameへ水平link[-35,0,−140]→[5,0,−140]と軸[5,0,−140]→[5,0,−170]を追加。固定yawCaseはscene直属の灰色半透明必要空間（r14,H22,中心[5,0,−157]）。manifestへ底部body/光束/外装検査を統合。公開は完成機構でなく必要空間の試作表示。
既存tab9をreloadして幅260奥行260高さ177表示とオーキャンv2稼働を確認、支持checkbox onでスクリーンショット確認。全job終了pendingなし。前回/今回ともprogress。次は光学rig（head+mirrors+projector）を前側roll軸へつなぐ支持/固定、底部固定・軸受肩/保持・駆動/強度・配線/放熱/首カバー。実瞳/合焦/光量/迷光の物理検証も未完。重量質問未回答。goal active。


### 光学rigをroll軸へ接続する経路調査
前回は底部v6公開反映を完了したprogress。今回はhead/鏡/projectorを実際にroll軸へつなぐ支持に着手。首近傍のhead/face頂点範囲を取得。頭奥へ支持材を伸ばす場合は、折返し鏡間だけでなくmain→faceの光を含める必要があるため、全original外面face頂点+full main mirror frontから反射後光束の凸包を生成。reflected-beam-hull.json（入力86616、凸包15431頂点30858面、minZ47.38885）。全鏡面の未使用領域も含む保守的包絡。
head-entry-support-screen.mjsで仮rod半径3、始点[45,0,0]→[45,0,20]→[entryX,0,20]→[entryX,0,100]を5案検査。entryX0は折返し光束（x0.5,z20）、10は折返し光束（z69）、20は反射後光束（z60.5）、25はhead壁（z49.5）、30はhead壁（z39.5）で隙間不足。これらを採用しない。首壁への固定接点が未定で100mmまで伸ばした探索なので、頭内部への全支持方式の不可能性ではない。次はbeamへ入る前の首元に接点を求め、頭固定と鏡/projector固定を分けて接続する。
公開v6は未変更。全処理終了pendingなし。鏡→face包絡を以前の首/底部検査へ自動適用したと主張しない（以前の部材は低い位置にあるが必要に応じ追加検証）。重量未回答。goal active。


### 首下端でのhead固定候補
前回は頭奥への5直線案を棄却したprogress。今回neck-contact-candidates.mjsでhead-wall表面の候補を探索。採用検討点は[16.9482,−35.0687,42.6266]と[17.2152,35.5275,42.5726]。反射後包絡から21mm程度離れる。元head頂点/外面を変更していない。
neck-fork-screen.mjsでX rollshaft[12,0,0]→[45,0,0]（r4）、[45,0,0]→[45,0,20]と左右接点の6mm下までの分岐（r3）を検査。中心線0.5mm刻み+.2501guard後、管外面からfold-main光束13.3599mm、main-face光束18.3243mm、本体35.6456mm、head壁2.47221mm、face壁18.3243mm。全rigが一緒に回るのでこれら相対距離は姿勢不変。実瞳/散乱/未計画迷光を保証しない。
先端は接点の6mm下で止めており、パッド/クランプ未設計。headを実際に支持したとは扱わない。次はforkと既存首機構/外装の相対roll/pitch干渉、接触パッド、projectorとmirrorの取付。公開v6は未変更。全処理終了pendingなし。重量質問未回答。goal active。


### neck forkの機構相対干渉検査
前回はforkと両光束/head/bodyの相対距離を確認したprogress。今回はneck-fork-outer-clearance.mjsを実行。fork vs公開outer7線分（底部含む）、pitch/roll1度+連続補正で管外面最小52.0511mm。forkの全姿勢包絡球radius55.4306、中心[5,0,0]は既存開口XY内・床より上なので外装を通過できる。
neck-fork-inner-clearance.mjsは有限円柱同士の最近接面距離でinner部材とのrollを評価。初回は保守的に切り欠き前のfront barを使ったため、X rollshaftとbarの交差を検出（実表示はcase用にy±14で切り欠いている）。実表示と同じsplit geometryへ修正しsession57899で再計算中。これが次回poll対象、再起動しない。初回のbar干渉は現行設計の欠陥と断定しない。自分自身のroll軸受ケースとshaftの意図した内包は除外。他部材との距離は検査対象。
公開v6は未変更、fork/padsはまだ表示していない。固定パッド未設計。goal active。


### 公開更新：neck fork候補
57899正常終了。実際のsplit U字に合わせたinner cylinder検査で、X shaft対pitch shaftの最小下限2.96148mm、fork上昇管対roll case3.83087mm、左右branch対roll case5.568/5.587mm。自分の軸受に入るshaftだけは意図した接続として除外。端点/軸方向位置が各他円柱の外部であることと正距離により全内包を除外できる。負荷/軸受適合は未確定。
neck-pad-surface-mapで首下端を12×12mm範囲・0.5mm格子で下から測定。左右の下向き法線セル面積推定55.25/56.75mm²、近傍有効投影面積83/81.5mm²。高さ約42.25..47.85で非平面。4角サンプルの推定で厳密接触面/耐荷重の証明ではない。次は形状適合padと保持方法。
公開assets/neck-fork.jsonとrig配下の表示を追加。首支持checkboxでforkも表示、先端は頭の下で止まりpad未設計と明記。外装寸法v6維持、manifestへfork検証を追加。既存tab9reload、ライブ顔映像・fork表示スクショ確認。全process終了pendingなし。前回/今回ともprogress。次は首pad/保持、projector/mirrorのrig固定、駆動/強度/冷却/首覆い・実機光学。重量質問未回答。goal active。


### 首パッド形状試作（未公開）
build-neck-pads.mjsで下向きsurface-mapセルの最大連結領域から左右padを生成。底面はfork tip Z、上面はhead underside0.5mm格子の補間−0.2mm。左右投影面積55.25/56.75mm²、各連結1成分、辺次数異常0。work/torso-flat/neck-pads/pad-0.stl / pad-1.stl、bin、manifest。
audit-neck-padsの独立三角形距離/交差検査でhead/face/body/両光束との交差0。head最小隙間左0.112976/右0.067016mm、反射後光束16.3929/16.0919mm以上。実造形誤差/接触圧/接着/機械保持は未確定。公称0.2mmは垂直方向の設計値で最短面距離と同じではない。
audit-pad-connectionsでforkとの意図した重なりとbeam/body凸包の非内包を検査（session27231）。頭/顔の非内包はpad底面が両shell最小Zより下で、連結padと対象表面が交差しないことから判断可能。次はpadの機構相対動作/外装、保持方法・造形適合の検討。公開v6+forkは未変更。前回/今回ともprogress、goal active。


### 公開更新：首pad試作
neck-pad-motion-audit.mjsでpad全三角形を包む球（半径8.4974/8.6764）を使用し、inner/outerフレームの円柱カプセルと相対pitch/roll全範囲を補正付き検証。inner最小下限17.0709/16.7801mm、outer47.5882/47.0056mm。支点中心の全姿勢包絡半径62.8718/63.3597mmは既存開口内・床より上で、外装を通過。forkとの意図した接合は別検証済み。
公開assetsにneck-pad-0/1のbin/STLを追加、headFork groupに金色padを追加しheadと同じrig変換を使用。manifestへpad形状/fit/connection/motion監査を統合。表示は実物保持・はめあい未確認と明記。既存tab9reload、支持表示on、pad表示スクショとライブ映像稼働確認。外寸v6維持。
前回/今回ともprogress。全process終了pendingなし。次はpad保持/締結とprojector・mirrorをroll rigへ固定する経路（まだ表示上は浮いている）。駆動/重量/強度/配線/冷却/首カバー・実機光学も未解決。重量質問未回答。goal active。


### projector固定の公式CAD調査
前回は首pad公開まで実施したprogress。今回はprojectorをroll rigへ固定する検討へ進む。Tripo calibration.jsonを再読：135×75は底板のみ。AI全体形状は近似でレンズ基準や穴は厳密CADではない。公式底板STEPを発見：work/internal-projector/ti-cad/DLPDLCR3010EVM-G2_Mechanical/DLP002_DLPDLCR3010EVM-G2-ENGINE-BASE-SS.STEP。assembly treeで底板bounds[0,0,0]..[135,3.97,75]を確認。
extract-base-features.pyで単体STEPの円柱面を抽出（base-cylinder-features.json）。r3の半円候補4面、r1.2/r2.2の段付き候補等があるが、円柱面はフィレットも含み、ねじ規格や空き固定穴と断定しない。単体座標は組立座標と異なる。
既存OCP環境work/internal-projector/cad-venv/bin/pythonが利用可能と確認（インストールなし）。extract-assembly-base.pyをsession3191で実行中。公式REVD assemblyから該当baseをloc変換付きで読み、円柱の角度範囲/face orientationを抽出してbase-assembly-features.jsonへ出力予定。次回同handleをpoll、勝手に再起動しない。全体本体と基準合わせして固定経路を検討するための情報。
公開v6+neck padsは未変更。重量未回答。goal active。


### projector底板の固定候補の解釈と登録
3191正常終了。base-assembly-features.jsonに公式REVD組立座標の円柱面/範囲/向きを保存。単体→組立変換は(X,Y,Z)→(−Z,Y,X)を数値確認。半径1.2/2.2で各2πの段付き穴はCAD[13.742,0,48.0002]と[61.742,0,48.0002]、穴間48mm。assembly treeで両方に92005A023_METRIC PAN HEAD PHILLIPS MACHINE SCREWが既に配置されているため空き固定穴として扱わない。半径3の4円柱は各π/2、深さ約1.6606mmで浅いくぼみの角。単純な貫通長穴という解釈は撤回。
全664532 kit頂点のCAD/local差を確認し、lens datum[111.62,13.97,81.335833]の平行移動に8e-6mm以内で一致。projector-base-anchors.mjsで公式底面4隅と6mm下面側の支持候補6点を現在の投影配置へ変換。projector-base-anchors.json。これは穴や荷重保証のある位置ではなく受け台設計の候補。
次は裏側受け台からroll shaftへつなぐ経路、クランプ/ねじの扱い・基板/ファン/光路との干渉。既存ねじを取り外して流用する案は未採用。公開v6+fork/padsは未変更。全process終了pendingなし。前回/今回ともprogress、重量未回答、goal active。


### projector底板支持経路の候補探索
前回は公式baseとoccupied fastenersを確認したprogress。今回はprojector-support-route-screen.mjsで始点[45,0,0]（roll軸前端）から6候補底面支持中心へ半径3mm棒材を1折れで接続する探索。中継grid x−80..120,y−100..100を20mm刻み、z−20/−60/−100/−140。全本体凸包とfold-main光束に中心線2mm以下+1.0001mmguardで通過する候補を各72〜81件得た。最短152.197〜208.780mm。reflected/headは全経路z<=0より上。
最初のprojector→fold光束は別検証とし、projector-support-first-leg-audit.mjsで各anchor上位5、計30経路を仮8mm機械レンズ面瞳→full fold mirror包絡に対して検査。全件正の距離下限（最小3.5371mm）。実射出瞳の位置/寸法の不確定性をなくすものではない。
anchor1最短例:[45,0,0]→[20,60,−100]→[42.5336,37.3412,−107.9417]。anchor3:[45,0,0]→[0,40,−100]→[8.8032,3.3588,−93.5382]。全経路は首フレームとの相対roll/pitch、動作外装、受け台・保持・荷重未検証。現在の公開模型へは追加していない。次はこの30候補の機構/収納評価から同時使用できる支持構成を選ぶ。全process終了pendingなし。重量未回答、goal active。


### 支持経路の首動作・収納検証
projector-route-motion.mjs/jsonで30経路を有限円柱の首フレーム・軸受体積に対して評価。空間2mm以下と角度区間の距離guardを減算。anchor1 rank2の1本は内側距離下限1.302896mm、外側7.894528mm、側面・底面収納を通過。経路[45,0,0]→[40,80,-100]→[42.533585,37.341222,-107.941665]、棒半径3mm、長さ171.625680mm。負の下限の他候補は不成立の証明とは限らず追加検証を要する。
projector-support-roof-audit.mjs/jsonも正常終了。5度角度セルの端点凸包に回転補間誤差と棒半径を加えた上面スラブ断面はX14.5586..58.5735,Y−33.2760..51.2198mm。現在の開口X−109..87,Y−125..125mm以内、passes=true。projector-support-selected-route.jsonに結果参照を追記。
現在の胴体外寸は幅260×奥行259.526×高さ176.860mmで変更なし。支持アーム1本の候補が収まることを検証した段階であり、受け台・脱落防止・ミラー固定・駆動・冷却・強度を完成したものではない。公開模型には追加していない。次は底板下面で複数支持点をつなぐ受け台と保持方法を検討。全process終了pendingなし、progress、goal active。顔ランドマークの補正より胴体の小型化を優先。


### 底面受け台の骨組み候補
前回turnは支持アームの上面開口検証を完了したprogress。今回はprojector-cradle-rails.jsonの7辺（底面6mm下面、半径3mm）をprojector-cradle-motion.mjs、projector-cradle-optical.mjs、projector-cradle-first-leg.mjsで検証。全辺はfull body凸包とfold-main包絡から距離下限2.736mm以上、仮D8のfirst-leg包絡から18.845mm以上。5辺2→4、1→3、3→5、4→5、2→3は首機構連続角度guardと外装側面/底面を通過。最小outer距離下限3.59966mm。これらはanchor1支持アームへ連結する骨組みとなる。0→1と0→2はouter下限負で採用せず（実交差確定ではない）。projector-cradle-selected.json保存。底板との接触pad、保持、強度、冷却は未検証、公開変更なし。全process終了、goal active、次は接触部と保持を詰める。


### 受け台から底板への接触支柱
前回は骨組み5辺選定のprogress。今回はanchor1..5に半径3mm・高さ6mmの底面法線方向支柱を定義（projector-contact-posts.json）。projector-contact-motion.jsonで全5本の内外首機構guard距離が正、外装側面/底面内。projector-contact-optical.jsonとfirst-leg.jsonでfold-main最小20.1263mm、仮D8 first-leg最小15.63265mmの距離下限。機械レンズ面仮瞳に依存。
projector-contact-footprint.pyは公式REVD assemblyから底板を組立座標に変換。CAD(X,Z)=(15,65),(65,10),(65,65),(120,10),(120,65)でY=0接触円盤R3、厚0.05mmの底板内層についてCAD差体積0、各128点も内側。Y−6..0支柱との底板共通体積0、全boolean正常。projector-contact-footprint.json。底板以外の組立部品・接触剛性・実物の公差・保持はまだ未検証。接触するだけでは首傾斜中に脱落を防げないため完成固定とは扱わない。公開変更なし。次は全CAD組立との支柱/骨組み干渉、および側面保持の検討。全process終了、goal active、progress。


### 接触支柱の全公式組立検証・受け台候補の公開
前回turnは接触支柱の底板CAD/首動作/光束検証を完了したprogress。projector-contact-assembly.pyで公式REVD STEPのfree root組立shapeと5本のCAD R3,H6支柱の共通体積を計算。全5本booleanDone=true、intersectionMM3=0。projector-contact-assembly.json。底板接触は意図したもの。保持・強度・実物公差の証明ではない。session74481正常終了。
公開assets/torso-flat-study/projector-cradle.jsonに選定支持アーム2辺、下面骨組み5辺、接触支柱5本を保存。neck-study.jsに紫色のprojectorCradleをrig子として追加し、機構checkboxに連動。HTMLに候補・未保持/強度未確認を明記。公式assembly支柱結果も同assetsへコピー。node --check通過。既存browser2 tab9 reload後、候補を表示し、0/0/0と45/20/15で描画・rig追従を視認。オーキャンv2正常。最後にreset、機構表示true、markDeliverable。
胴体外寸変更なし。全顔のchief率255542/255542は従前条件の値で、受け台について別途保守光束距離を検査済みだが実機100%とはしない。次は保持構造、鏡支持、駆動/冷却等。全process終了、goal active、progress。


### 底板上側の爪・コの字保持具の探索
前回turnは接触支柱の全組立干渉と受け台公開のprogress。今回は公式REVD assemblyでtop-lip-screen.pyを実行。CADX65のZ0/70上爪6×3×5（Y4.07..7.07）は共通体積0。X120は29.60198/3.85302mm3の交差、採用不可。光路/動作が通ってもCAD部品干渉で落ちる例。projector-top-lip-screen.json完了。
X65に限定しC型保持具を上下/側面3boxで定義（projector-edge-clamp-screen.py/json）。Z0側全形状は全公式組立との共通体積0。Z70側は2.6517697mm3の交差で不採用。各パーツの個別識別は次。両側の保持具が完成したとは扱わない。中央の対向クリップだけでは長手方向滑りも未解決。
projector-edge-clamp-lines.mjs/json、motion.mjs/jsonでboxを包むcapsuleを評価。motionの半径は底sqrt18、側/上sqrt11.25。内roll1°、外pitch/roll1°nearestguardで全正、外最小3.05376mm。外装側/底内。光束は全boxをR4.25で保守評価、fold-main最小30.30105mm、仮D8 first-leg最小29.55096mm。これら合格はZ70側の実CAD交差を覆さない。公開表示は以前の受け台のまま（保持具未追加）。17342/22118正常終了、pendingなし、goal active、progress。次はZ70側3boxを個別に調べ、干渉箇所回避と長手方向止め/組付け方法を詰める。


### 保持具干渉の部位特定・v2
前回はC保持具screenのprogress。projector-edge-clamp-interference.pyでZ70側3boxを個別に検証。下面0、側面44.9487856、上面21.4583925mm3。以前の重複box compound共通体積2.65177は定量値として不適切、個別結果を優先する。候補不採用の判断は維持。
v2は下面[62,-9,62]+[6,6,17.8]、側面[62,-9,76.8]+[6,15.57,3]、上面[62,4.07,70]+[6,2.5,9.8]。側面を1.7mm外へ、上面厚さ3→2.5mm。projector-edge-clamp-v2-interference.json完了(session26629終了)：下面0、側面0、上面16.186071mm3、干渉bounds[63.953988,4.07,75]..[68,6.57,76.6002]。v2も不採用。上の跨ぎ部に逃げが必要。
重複assembly compound処理の定量不確かさを避けるため、各transform済leafと個別boxのbooleanを行うprojector-edge-clamp-v2-leaves.pyを追加。session58044が現在実行中、直近pollはRead RetDone、Transfer待ち。次回同sessionをpollし、projector-edge-clamp-v2-leaves.jsonを読む。再起動しない。候補boxとleafのCAD boundsで保守broadphase。全root探索。公開は受け台のみのまま。首動作/光路検証は旧保持具のものでv2へ流用しない。goal active、progress。


### 個別leaf干渉結果と保持具v3〜v5
前回progress＋live58044をpollして正常終了確認。v2上爪はMAIN-BOARD-MOUNT単体と16.186071mm3干渉。下面/側面は各leafの保守bounds候補なし。
v3はX57..63へ5mm移動、下面X57..68で既存anchor3に接続。projector-edge-clamp-v3-leaves.json：上爪がSHANE-PRT0016/17/19と3.17964/2.90131/3.17964mm3干渉。MAIN-BOARD-MOUNTは0。v3不採用。motion/opticsは通過したが実CAD干渉を覆さない。
CADtreeのY4.07..6.57,Z70..79.8の部品boundsを調査。X13.223..61.224には連続したSHANE部品、X74以上はengine等。X8..13候補をv4として下面[8,-9,62]+[10,6,17.8]、側面[8,-9,76.8]+[5,15.57,3]、上面[8,4.07,70]+[5,2.5,9.8]。v4-leavesでは上面MAIN-BOARD-MOUNT11.969448mm3。v4不採用。
v5はこの支持部品をまたぐアーチ：下面v4同一、外柱[8,-9,76.8]+[5,39.3,3]、上橋[8,27.8,70]+[5,2.5,9.8]、内爪[8,4.07,70]+[5,26.23,4.7]。projector-edge-clamp-v5-blocks.json等。leaf干渉：前三部品はbounds候補なし、内爪がSHANE-PRT0004と23.571243、MAIN-BOARD-MOUNTと37.6047mm3、v5も不採用。首動作全正/収納内、fold-main最小81.5107、仮D8 first-leg最小80.7606mmでもCAD不成立。
scriptセットは各v3/v4/v5のlines,motion,optical,first-leg,leaves。注意：コピーされたscope文章の固定半径表記は古い箇所あり、実装は各box最長軸のcapsule半径radiusMMを使用、motion外1°/inner1°guard。脚部は三角体積等にせずbox保守。全session58044,61018,75476,20569正常終了pendingなし。公開は受け台のみ、保持具未追加。goal active、progress。次はMAIN-BOARD-MOUNTの断面/占有領域を見て上爪の通過可能性を直接調べるか、既存の底板段付き穴と既存ねじを使う固定案を具体的CAD寸法で再検討する。現在のねじは空き穴として扱わない。胴体変更なし。


### 既存底板ねじを共締めする案へ
前回turnは保持具v3〜v5の不成立を確認したprogress。今回CADtreeとbase-assembly-featuresを再読、底板の48mm間隔段付き穴(CADX13.742/61.742,Z48.0002)に実ねじinstance18/19が占有することを確認。projector-existing-screw-features.py/jsonで当該2ねじの変換済円柱を抽出。半径1の外径、0.4mm周期のねじ形状、先端Y8、座面Y2の首下6mmに整合。web92005A023検索はメーカーの正確な製品仕様を確定できず、購入型番確定には使っていない。
下面固定ボス案：R4,H10,Y−10..0、貫通R1.2、下面座ぐりR2.2深2（座面Y−8）。既存ねじ首下6から+10=16mmの交換候補で先端Y8を維持する寸法案。現在のねじを空き穴扱いせず、交換を前提とした候補。threads/締結強度/工具アクセス未検証、実部品/公開CADのねじは変更なし。projector-screw-boss-leaves.py/jsonで各boss対個別leaf:底板/MAIN-BOARD-MOUNT体積0、その他保守bounds除外。
projector-screw-boss-lines.mjs/jsonでboss外包R4円柱と、Y−6の穴中心からanchor1/3(X15/65,Z65)へのR3接続棒を定義。motion.mjs/jsonはouterpitch/roll0.25°guardへ細分化、inner1°。全正、outer最小1.444192mm、側/底収納可。初回1°下限−0.202は未証明であって実交差とはしていない。optical/first-leg検証全正、fold-main最小39.96887mm、仮D8 first-leg39.21878mm。接続棒のfullbody凸包クリアランス、接続箇所・工具アクセス、座面の実接触・thread適合・強度は次。公開変更なし。20767/76210/43074正常終了、pendingなし、goal active、progress。


### 一体受け台CAD生成・ねじ通路・表示更新
前回はねじboss案のCAD/動作/光路検証progress。projector-screw-connector-clearance.mjs/jsonでR3接続棒2本のfull body凸包距離下限2.73778/2.73637mmを確認。
projector-cradle-cad-lines.mjs/jsonで公開旧骨組みと5接触支柱、2接続棒を公式CAD座標に逆変換。build-screw-cradle.pyで全円柱とR4boss2個を融合し、ねじ貫通R1.2と下面座ぐりR2.2深2を最後に切除（接続棒が穴を塞がないようにする）。work/torso-flat/screw-cradle/cradle.step/stl,audit.json。valid=true、solidCount1、volume13943.9582mm3。CAD座標Y−60..−8.1のR2工具軸と受け台の共通体積は両方0（取り外した受け台のみ、実工具/組立全体/ねじ頭スロット適合は未保証）。16139正常終了。
公開assets/torso-flat-study/projector-screw-cradle.stl/.step/-audit.json追加。projector-cradle.jsonのcadMesh/cadDatumで表示指定。neck-study.jsはSTLLoaderでCAD原点からdatumを引き、projector位置/回転を適用した紫色meshをrigへ追加。旧円柱描画は置換。ねじ交換・適合未確認をUI注記。JS構文チェック通過。browser2 tab9 reload、機構表示、0/0/0と−45/−20/−15で描画と追従確認、ライブ映像正常。最後reset・機構true・markDeliverable。
まだ元のprojector Tripo本体に見える既存ねじ形状は交換表現にしていない。交換ねじの具体品/ねじ込み先/締結荷重/座面/工具アクセス（装着状態）とroll shaft端への保持は未検証。ミラー固定、駆動、冷却も残る。胴体外寸不変、実機全顔100%未証明。全process終了pendingなし、goal active、progress。


### 交換ねじメーカー寸法と受け側
前回turnは一体受け台生成・公開のprogress。projector-thread-receiver-features.py/jsonでMAIN-BOARD-MOUNTとFAN-MOUNTのねじ軸近傍円柱を抽出。相手はFAN-MOUNT-SS、各軸R0.8、CADY3.9738..5.9738の深2mm（単純円筒、雌ねじ形状は未再現）。M2ピッチ0.4なら名目5ピッチ区間だが実噛合い/強度ではない。46016正常終了。
Aspen公式図面https://www.aspenfasteners.com/content/2D_PDF/product61/ME031-204X16.PDFをwebで確認。ME031-204X16はM2−0.4×16、全ねじ、A2、Phillips H #1、頭径最大4mm、頭高1.60±0.12mm。座ぐりD4.4深2は名目半径隙間0.2、最大頭高余裕0.28mm。新座面Y−8+首下16で旧先端Y8と名目一致。長さ公差・実雌ねじ・材質/引抜強度は未確認。元はファン支持部の締結で、ロボット動荷重適合は保証できない。projector-fastener-fit.jsonをwork/publicassetsに保存。購入/取り付けは行っていない。表示形状変更なし。次は座面荷重経路・締結への動荷重を整理し、ミラー支持とroll軸固定等の未設計箇所を詰める。全process終了pendingなし、goal active、progress。


### 小鏡支持アーム候補
前回turnはメーカーねじ図面/相手穴の確認progress。今回は未支持のfold mirrorへ進む。fold-mirror-support-screen.mjs/jsonは受け台肘[40,80,−100]から小鏡背面中心−8*normalまでR3棒を探索、fullbody凸包とfold-main包絡に空間2mmguard。224経路、直線119.980234mmが最短。fold-mirror-support-first-leg.mjs/jsonで仮D8 firstleg下限3.9999007mm。motion.mjs/jsonは首機構continuousguard正（inner44.84568、outer23.50271mm）、側/底収納可。selected.json保存。鏡厚み・接触/保持/調整・部材剛性未検証、公開未追加。
探索中の始点=waypointで長さ0辺が生じ、コピーしたfirstleg/motionのn=ceil(length/2)が0でNaNになったため、今回2scriptをmax(1,ceil(...))へ修正して再実行、全5候補有限値。directrank0の結果は変化なし。重複経路rank1は同一解で独立根拠にはしない。旧検証のnonzero経路に影響なし。72693正常終了、全process終了pendingなし、goal active、progress。次は鏡厚み/背面受け/固定方法を加えた候補を検証して可視化する。


### 小鏡厚さ・裏面受けの検証と表示
前回は直線小鏡支持候補探索progress。今回は小鏡の仮厚2mm（前面反射面0/裏面−2）、接着余裕0.2mm、裏面disk R5 Z−4..−2.2、stem R3 Z−8..−4を定義。fold-mirror-backing-lines/body/motionおよびfold-mirror-thick-lines/motionを実行。fullbody凸包から受け距離下限14.6052mm以上。内/外首機構capsuleguard正、厚み入り鏡も内28.3412/外58.0586mm。鏡capsuleを含め動作外装側/底内、maxZ−5.5374以下で屋根内面−0.1397より下。
fold-mirror-thickness-audit.mjs/json：全2mm厚小鏡箱とfullbody凸包surface距離3.333934mm、鏡頂点のbody内包false（面交差/鏡内へのbody全包含はbox小サイズとbody全体extentで除外、厳密絶対公差証明ではない）。仮8mm瞳とmainmirror全頂点の小鏡表面normal半空間距離最小3.00942/52.82367mm。対応するconvex光束は表側、disk/stem全体は裏側にあり予定first/fold-mainを遮らない。異常迷光/実射出瞳は未保証。
公開fold-mirror-support.jsonに支持直線とdisk/stem、厚2を保存。neck-study.jsのprojectorCradleに紫色支持円柱・裏面箱追加、機構toggleに連動。HTMLに仮厚2、接着/調整/保持未検証を明示。syntax通過、browser2 tab9 reload/AX/スクリーン確認、live映像正常。最後0/0/0、機構true、markDeliverable。cradle.step/stlは小鏡支持追加前の受け台のまま、混同しない。元鏡の反射前面/位置は不変。全process終了pendingなし、goal active、progress。次は小鏡受けの剛性/調整/保持、主鏡の支持、軸固定等。


### 主鏡の厚み検証開始
前回turnは小鏡厚み・受けの検証と公開を完了したprogress。今回main-mirror-thickness.mjsを追加し、974点trimmed輪郭、反射前面固定で裏側へ0.2/1/2/3mm押出し、head/face内壁meshとの最近接面距離を比較。寸法129.8459×155.4032mm。surface距離のみでmaterial内包は別検証を要する。
実行session91754が現在live、直近write_stdin pollはsession返却・未出力。未完了/エラーではない。次回同sessionをpollしmain-mirror-thickness.jsonを読む。再起動しない。mirror querygeometryにboundsTree未設定のため低速の可能性があるが、計算自体の結果は無効ではない。将来の別実験で高速化するならgeometryへboundsTree追加を検討。公開変更なし。goal active、現在specific live process待ち。


### 主鏡計算高速化・裏面支持点探索
前回turnは主鏡厚み計算開始とlive91754待ち。今turnでsamehandleがliveと確認、psでPID38735(経過3:10 CPU73.9%)を観測。実装のquery mirror geometryにboundsTreeがなくclosestPointToGeometryの低速fallbackが使われるとライブラリsourceで確認。性能上の原因を修正するため意図的にSIGTERM、91754 exit143確認後、鏡geometryにMeshBVHを追加し同一形状/条件で再実行。単なる観測timeoutを理由とする再起動ではない。
現行session96324 live。直近出力はthickness0.2/head surfaceDistance1.3244826584mm、witness[40.4332454,0.8138051,252.8988644]。faceおよび1/2/3mmは続行中。次はsame96324をpoll。jsonは各厚のhead+faceが終わってから保存する仕様。
並行してmain-mirror-backing-clearance.mjs/jsonを作成。局所u,v各−40/−20/0/20/40、backdepth5/8/12、75点全てhead最近接面距離>3mmかつ前面から裏点への法線segmentがheadwallを横切らない。R3sphere候補として空間あり。最良u0v0depth5距離44.3666mm。全体のcavity所属/face干渉/保持/剛性は独立未証明で、mirror全厚が通るという意味ではない。公開変更なし。前回verified wait、今回progress。goal active、pending96324のみ。


### 主鏡厚み結果・三角裏枠
前回progress（計算高速化とhead側支持点）。今回same96324をpollして全結果/exit0を確認。main-mirror-thickness.json：厚0.2/1/2/3mmでhead最小表面距離1.324483/0.884861/0.327705/0mm。3mmは接触交差、採用不可。全厚face距離17.694081mm。2mmは名目非接触だが実物公差の余裕が小さい。主鏡前面・輪郭・公開形状は変更なし。
main-mirror-backing-face-clearance.mjs/jsonで75sphere候補がfaceに対しても距離>3、frontからbackへの法線経路非横断。main-mirror-back-frame.mjs/jsonで鏡local[-20,-20,-8],[20,-20,-8],[0,20,-8]の三角R3部材を定義。0.5mm空間サンプル+0.25001guardでhead/face面隙間正。未接続候補で、鏡/頭への取り付け・強度・光束全体・material内包判定は別途。支持枠公開未追加。
次は主鏡実厚と公差余裕：反射前面を保った背面縁形状（例えばbevel、製作可能性未確認）または許可範囲内の内壁余裕を比較。3mmそのまま合格としない。既存の表面距離だけで実機収納完成とはしない。全process終了pendingなし、goal active、progress。


### 2026-09-22 小型男性胸像への外観変更
ユーザーが幅430mmの筋肉質案を大きすぎ・人体に見えないと指摘。頭部は変えず、上腕を省いた短い胸像へ更新。新外形330×211×220mm（幅×奥行×高）、旧430×336.55×278mm比で幅23%、奥行37%、高さ21%減。断面を楕円に変更、胸筋・胸骨溝・鎖骨・上背部の起伏、首へ傾斜する肩を生成。寸法は造形目標で成人の実測値ではない。
work/buildable-bust-v1/generate-wide-backup.mjsに直前ジェネレータ保存。現行generate.mjsから8分割STL生成、全8部品で辺接続数2、仮定220×220×250mm以内。ただし継手・ポート・正常厚み・可動干渉は未検証。buildable-bust-study.htmlを再読込しスクリーンショットで確認。旧モーター寸法枠が底より突出し、旧機構の収納は成立していない。外装縮小だけで達成とは扱わない。プロジェクタや光学系を縮小していない。
暫定モーターXM430-W350-T：28.5×46.5×34mm、82g、12V、停動4.1Nmは連続許容値ではない。公式 https://en.robotis.com/shop_en/item.php?it_id=902-0124-000 。質量・重心・連続トルクが未確定のため購入推奨確定前。TI G2は19V電源案、12Vサーボと分離。HDMI/DC背面パネルは位置表示のみ。
次工程は小型胸像へ実寸プロジェクタ・ケーブル・モーターを再配置し干渉を解消。主鏡厚2mm公差不足、可動首クリアランス、放熱、接合設計を継続。goal active、完成扱い不可。
