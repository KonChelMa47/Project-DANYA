import json,csv,math,shutil
from pathlib import Path
root=Path('/Users/daniilmalchenko/Documents/Codex/2026-09-21/bl');out=root/'outputs/danya-simulator/assets/buildable-bust-v2';s=json.loads((root/'work/torso-flat/torso-v3-layout.json').read_text())
for key in ['mirror','foldMirror']:
 m=s[key];pts=[(x*m['width'],y*m['height'])for x,y in m.get('outline',[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])];d=['0','SECTION','2','HEADER','9','$INSUNITS','70','4','0','ENDSEC','0','SECTION','2','ENTITIES','0','LWPOLYLINE','8','REFLECTIVE_APERTURE_NOT_CUT_LINE','90',str(len(pts)),'70','1']
 for x,y in pts:d+=['10',str(x),'20',str(y)]
 d+=['0','ENDSEC','0','EOF'];(out/f'{key}-optical-aperture-mm.dxf').write_text('\n'.join(d)+'\n')
 w,h=m['width'],m['height'];poly=' '.join(f'{x+w/2+10},{h/2-y+10}'for x,y in pts);(out/f'{key}-optical-aperture-mm.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w+20}mm" height="{h+20}mm" viewBox="0 0 {w+20} {h+20}"><polygon points="{poly}" fill="none" stroke="black" stroke-width="0.2"/><text x="10" y="6" font-size="3">OPTICAL APERTURE ONLY / mm / NOT FINAL CUT LINE</text></svg>')
rows=[]
for mass in [1,1.5,2]:
 for com in [20,40,60]:
  demand=2*(mass*9.80665*com/1000+.03*.5+.05)
  for ratio in [1,2,3]:rows.append(dict(movingMassKG=mass,COMOffsetMM=com,ratio=ratio,designDemandNM=demand,estimatedOutputNM=.82*ratio*(1 if ratio==1 else .8),screenPass=demand<=.82*ratio*(1 if ratio==1 else .8)))
with (out/'motor-torque-screen.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
bom=[
['F01','M3×20 六角穴ボルト','16+予備4','外装試作用','3.4mm穴試験片の確認後。モーター用ではない',''],
['F02','M3 六角ナット（対辺5.5mm）','16+予備4','外装試作用','薄いPETGを強く締めない。内部から工具で保持',''],
['F03','M3 平座金（外径約7mm）','32+予備8','外装試作用','両側に1枚ずつ。穴・接合部への収まり実物確認',''],
['P01','PETG 1.75mm','必要量はスライス後','外装試作用','白または好みの色。プロジェクタ周辺の耐熱は別試験',''],
['M01','ROBOTIS XM430-W350-T','3','選定候補・注文保留','12V TTL。重量・重心と連続動作試験で確定。正式CAD取得済','https://e-shop.robotis.co.jp/product.php?id=44'],
['C01','ROBOTIS U2D2','1','選定候補・注文保留','TTL版モーター用PC通信。電源供給機能は別','https://emanual.robotis.com/docs/en/parts/interface/u2d2/'],
['C02','ROBOTIS U2D2 Power Hub Board','1','選定候補・注文保留','12Vサーボ給電。定格電流・コネクタを電源と最終照合','https://emanual.robotis.com/docs/en/parts/interface/u2d2_power_hub/'],
['E01','12V外付けACアダプター','1','仕様検討','3軸ストール合計6.9A。5A品は電流制限設計なしでは採用しない',''],
['E02','既存プロジェクタ用19V ACアダプター','1','既存品を使用する方針','12Vサーボ系統とは別。DCプラグ寸法・極性を既存品で照合','https://www.ti.com/tool/DLPDLCR3010EVM-G2'],
['O01','主鏡：表面反射平面鏡・特注輪郭','1','発注不可・要再設計','有効輪郭129.846×155.403mm。3mm厚は頭内壁接触、2mm厚は隙間0.328mm。DXFは有効光学領域で切断線ではない','https://www.edmundoptics.com/p/200-x-200mm-Enhanced-Aluminum-4-6lambda-Mirror/46416'],
['O02','小鏡：表面反射平面鏡','1','発注保留','有効領域23.828×35.185mm。縁の保持代・角度調整機構を加えて再確定',''],
['W01','HDMI変換／延長ケーブル','1組','発注保留','実機端子種別・プラグ高さ・曲げ半径を確認。公式CADのHDMI-C表記のみでは確定しない',''],
['W02','DC延長・ストレインリリーフ・配線留め','1式','発注保留','プラグ抜き差し空間、可動ループ、放熱経路を確定後',''],
['D01','減速ベルト・プーリー・支持軸受','3軸分','未選定','重心・軸位置が未成立のためサイズと数量を確定しない','']]
with (out/'BOM.csv').open('w',encoding='utf-8-sig',newline='')as f:w=csv.writer(f);w.writerow(['ID','部品','数量','状態','条件','出典']);w.writerows(bom)
shutil.copy(root/'work/buildable-bust-v2/motor-cad-audit.json',out/'motor-cad-audit.json')
(out/'README.md').write_text('''# Project DANYA：外装組立試作 v2\n\nこれは無通電の外装組立試作パッケージです。駆動ロボットとしての発注・本番印刷リリースではありません。現状は可動時に実寸プロジェクタが外装へ干渉します。\n\n## 印刷\nA1は256×256×256mm、AD5Mは220×220×220mm。共通基準はAD5M、0.4mmノズル、0.20mm積層、PETG、4周壁、20%インフィル、5mm外ブリム、必要箇所にサポート。温度・冷却・流量は使用フィラメントと実機で調整。G-codeは別機種へ流用しない。\n- 最初は `M3-joint-coupon-hole-3.4.stl` を2個印刷。M3ボルトが手で通り、ナットと座金で割れずに締結できるか確認。きつければ3.6mm、緩すぎれば3.2mm試験片を比較。選択した穴径を本体へ反映してから本体印刷。\n- `shell-*-print.stl` が各部品をベッド原点側へ移したデータ。1部品ずつ、底面をベッドへ合わせて印刷。合計8部品。`assembly-coordinates.stl` は全体配置用で、そのままベッドに置かない。\n- メッシュ辺の閉鎖・一体性と造形範囲を確認済み。サポート除去、実物公差、締結工具アクセス、耐熱・強度は実物試験前。\n\n## 組み立て順（外装のみ）\n1. 上下各4部品を、内部の垂直接合部で仮組み。各リング4本、合計M3×20を8本使用。ナットとボルト頭の双方に座金を入れる。\n2. 上下リングを水平接合部8か所で合わせ、残り8本で接続。内部へ工具を入れるため底面を開いたまま作業。順番によって工具が入らない場合は最後の後側部品を外して作業し、アクセス不良を設計へ反映。\n3. 全部のねじを仮締めして外形を揃えてから手で軽く締める。樹脂接合部への指定締付トルクは未評価。\n4. 底板・機構・頭部はこの外装へまだ取り付けない。外装のねじ接合部を駆動機構の支持に流用しない。\n\n## 鏡\nDXFとSVGはmm単位の予定有効光学領域。切断発注図ではなく、面取り・保持代・製作公差を含まない。主鏡3mm厚は既存頭部に接触、2mm厚も約0.328mmしか余裕がなく本番不適合。表面反射鏡の特注厚みと固定部を再設計し、光束欠けを再検証する。\n\n## 駆動の計算条件\nXM430の4.1Nmはストール値。公式米国ページの0.82Nmはストール値の20%に基づく推定であり、連続動作保証ではない。CSVは質量1〜2kg、重心20〜60mm、慣性0.03kgm²、角加速度0.5rad/s²、配線抵抗0.05Nm、安全係数2で比較。減速機効率80%は仮定。実重量・重心・加減速・通電温度が確定するまでモーター発注を保留。\n\n## 残る設計\n可動時収納、支持軸と減速機、主鏡の厚みと固定、小鏡角度調整、映像の合焦と光量、HDMI/DC端子の実寸と抜き差し、配線の可動ループ、放熱、底板、転倒安定性、頭部との接続、非常停止後の頭部保持。これらを未検証のまま「印刷すれば完成」と扱わない。\n\n## 出典\n- https://us.store.bambulab.com/products/a1\n- https://www.flashforge.com/products/adventurer-5m-3d-printer\n- https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/\n- https://robotis.us/products/dynamixel-xm430-w350-r （推定トルク参考。採用候補はTTLのT型）\n- https://www.ti.com/tool/DLPDLCR3010EVM-G2\n- https://www.edmundoptics.com/p/200-x-200mm-Enhanced-Aluminum-4-6lambda-Mirror/46416\n''')
