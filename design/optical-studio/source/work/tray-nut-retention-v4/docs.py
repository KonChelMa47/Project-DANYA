from pathlib import Path
import shutil
r=Path('outputs/danya-simulator/assets/projection-prototype');v=r/'tray-nut-retention-v4'
note='''# 試作トレー v4 — M3ナット抜け止め

![トレー裏面](tray-underside.png)
![受け・ナット・カバーの分解図](nut-cover-detail.png)

青緑がトレーの受け、銀色がM3ナット、橙色が抜け止めカバーです。分解図のカバーは形状を見せるため浮かせています。実際は中央の通風窓側から横へ差し込みます。

## 印刷するもの

- `drop-projector-tray-t5.stl` 1個：旧トレーには溝がないため再印刷が必要です。
- `tray-nut-cover-0.stl` ～ `tray-nut-cover-3.stl`：各1個、合計4個。同形状です。
- `tray-nut-fit-coupon.stl` 1個：本体を印刷する前の嵌合確認用。組み込みません。

ZIPの `print/` をmm単位・100%倍率で印刷します。カバーはPETGを推奨、0.2 mm積層。薄い爪・レールがスライサー上で欠落していないことを確認します。組立位置のSTLとは区別してください。

## 組み立て

1. まず確認用パーツとカバー1枚を印刷し、M3六角ナット（対辺5.5 mm、厚さ2.4 mm）で試します。
2. トレーを裏返し、4か所の六角ポケットへナットを奥まで入れます。
3. カバーのレール・爪がある面をトレー側へ向けます。中央の通風窓側の入口に置き、外周側へスライドさせます。左右の爪が受けのくぼみに入り、中央のねじ穴がナットに一致するまで差し込みます。
4. 軽く逆向きに引いてカバーが戻らないこと、トレーを表返してナットが落ちないことを確認。途中で固くなる場合は無理に押し込まず、バリ・サポート跡・寸法を確認します。
5. 架台側から既存のM3×10とワッシャーを通して締結します。追加の金属部品や接着剤は不要です。
6. 取り外す場合は架台から外し、カバー左右の爪を内側へ押しながら中央の通風窓側へ引き戻します。

T形レールが下方向の脱落を、左右の爪が横方向の抜けを止めます。プロジェクタの受け寸法135.8×75.8 mm・取付ねじ軸・光学配置は維持しています。本番の可動機構はこの変更の対象外です。

## 検証範囲

CAD上でナットとカバーの抜け方向の係合、4か所の架台との食い込みゼロ、爪の弾性変形部分を除く挿入経路を確認しました。顔面167,698点の幾何光路で追加の遮蔽は0点です。STLの閉じた形状も検証しています。

実際の印刷公差、爪の差し込み力・保持力、繰り返し耐久性、プロジェクタ使用中の温度は未試験です。確認用パーツで保持を確かめてから本体を印刷してください。
'''
(v/'ASSEMBLY.md').write_text(note)
p=r/'ASSEMBLY.md';s=p.read_text().replace('投影試作 03','投影試作 04').replace('8ページ','9ページ').replace('組立部品28点＋鏡の実寸マスター2点＋磁石の穴径確認用パーツ1点','組立部品32点＋鏡の実寸マスター2点＋磁石・ナットの嵌合確認用パーツ各1点')
s=s.replace('の裏面4か所へM3ナットを入れ、架台4本と接続します。','の裏面4か所へM3ナットを入れ、`tray-nut-cover-0…3` を通風窓側から外周へ差し込んで爪をロックします。抜けないことを確認して架台4本と接続します。[抜け止めの図解・印刷順](tray-nut-retention-v4/ASSEMBLY.md)を参照。旧トレーは再印刷が必要です。')
p.write_text(s)
p=Path('outputs/danya-simulator/projection-prototype.html');s=p.read_text().replace('STATIC PROTOTYPE 03 / 後頭部接続一体型','STATIC PROTOTYPE 04 / トレーナット抜け止め').replace('8ページ','9ページ').replace('src/projection-prototype.js?revision=head-connectors-v3','src/projection-prototype.js?revision=tray-nut-retention-v4');s=s.replace('<details open>','<details open><summary>トレーのナット抜け止め</summary><p>横差しカバー4枚をレールと爪で保持します。旧トレーは再印刷してください。M3×10ねじはそのまま使用。まず嵌合確認用パーツで試してください。</p><a href="assets/projection-prototype/tray-nut-retention-v4/tray-update-print.zip" download>トレー更新用STLセット</a><br><a href="assets/projection-prototype/tray-nut-retention-v4/ASSEMBLY.md">抜け止めの図解と組立手順</a></details><details>',1);p.write_text(s)
p=Path('work/prototype-assembly-guide/build-pdf.py');s=p.read_text().replace('投影試作03','投影試作04').replace('STATIC PROTOTYPE 03','STATIC PROTOTYPE 04').replace('head-connectors-v3 /','tray-nut-retention-v4 /').replace('{page} / 8','{page} / 9').replace('組立部品28点','組立部品32点').replace('鏡型2点と穴径確認用パーツ1点','鏡型2点と磁石・ナット確認用パーツ各1点').replace("['後頭部・顔','2'","['ナット抜け止めカバー','4','tray-nut-cover-0…3'],['後頭部・顔','2'")
s=s.replace('先に入れます。架台側からM3×10','先に入れ、カバー4枚を横から差してロックします（9ページ参照）。架台側からM3×10')
shutil.copy2(v/'nut-cover-detail.png',r/'guide/nut-cover-detail.png')
s=s.replace('finish();c.save();', '''finish()
start('08 / トレーのナット抜け止め','改訂04：トレーを再印刷し、横差しカバー4枚を追加。M3×10ねじは変更なし。')
img('nut-cover-detail.png',44,396,507,302)
y=p('分解図：青緑＝受け、銀＝ナット、橙＝カバー。実際は横から差します。',383,10)
y=p('1. PETGで確認用パーツとカバー1枚を先に印刷し、M3ナットの保持を確かめます。旧トレーには溝がないため、カバーだけの追加はできません。',y,11)
y=p('2. 裏面の六角ポケットへナットを入れます。カバーのレール面をトレー側へ向け、中央の通風窓側から外周へ差し込みます。',y,11)
y=p('3. 左右の爪がロックし、ねじ穴が一致するまで差し込みます。軽く引いて戻らず、表返してナットが落ちないことを確認します。',y,11)
y=p('4. 架台側からM3×10＋ワッシャーで締結。外す時は架台を外し、左右の爪を内側へ押してカバーを通風窓側へ戻します。',y,11)
p('レールが下方向、爪が横方向の脱落を止めます。CAD上の係合・干渉・光路は確認済みです。実印刷での差し込み力、保持力、耐熱・耐久性は未試験です。',y,10)
finish();c.save();''')
p.write_text(s)
p=Path('work/projection-prototype/package.py');s=p.read_text().replace('len(audit)==31','len(audit)==36');s=s.replace("with zipfile.ZipFile(r/'projection-prototype-print.zip','w'", "files += [r/'guide/assembly-guide.pdf'] + [p for p in (r/'tray-nut-retention-v4').iterdir() if p.suffix in ['.json','.png','.md'] and p.name!='parts.json']\nwith zipfile.ZipFile(r/'projection-prototype-print.zip','w'");p.write_text(s)
