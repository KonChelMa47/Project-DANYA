from pathlib import Path
import json,math,shutil
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
R=Path.cwd();a=R/'outputs/danya-simulator/assets/projection-prototype';out=R/'output/pdf/DANYA-prototype-assembly-guide.pdf';pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'));W,H=A4;c=canvas.Canvas(str(out),pagesize=A4);c.setTitle('DANYA 固定式投影試作07 図解組立ガイド');c.setAuthor('Project DANYA');ink=colors.HexColor('#17373c');teal=colors.HexColor('#297d80');gray=colors.HexColor('#edf3f3');page=0

def p(txt,y,size=11,x=44,w=W-88,col=ink):
 st=ParagraphStyle('p',fontName='JP',fontSize=size,leading=size*1.55,textColor=col,wordWrap='CJK');o=Paragraph(txt,st);ww,hh=o.wrap(w,H);assert y-hh>(10 if size==8 else 30),(page,txt[:40],y,hh);o.drawOn(c,x,y-hh);return y-hh-9

def start(title,sub):
 global page;page+=1;c.setFillColor(teal);c.rect(0,H-12,W,12,fill=1,stroke=0);p('PROJECT DANYA / STATIC PROTOTYPE 07',H-30,9,col=teal);p(title,H-62,24);p(sub,H-103,10);c.setStrokeColor(gray);c.line(44,35,W-44,35);p(f'固定式試作専用 / prototype-v7 / 2026-09-24     {page} / 10',30,8,col=colors.gray)

def img(name,x,y,w,h):c.drawImage(str(a/'guide'/name),x,y,w,h,preserveAspectRatio=True,anchor='c',mask='auto')
def table(rows,y,widths):
 st=ParagraphStyle('t',fontName='JP',fontSize=9,leading=13,wordWrap='CJK',textColor=ink);data=[[Paragraph(str(t),st)for t in row]for row in rows];tb=Table(data,colWidths=widths);tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),gray),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#bdcece')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));_,hh=tb.wrap(W-88,H);assert y-hh>45;tb.drawOn(c,44,y-hh);return y-hh-15

def box(x,y,w,h,label,fill=gray,size=10):
 c.setFillColor(fill);c.setStrokeColor(teal);c.roundRect(x,y,w,h,4,fill=1,stroke=1);p(label,y+h-8,size,x+5,w-10)
def arrow(x1,y1,x2,y2):
 c.setStrokeColor(teal);c.setFillColor(teal);c.setLineWidth(1.4);c.line(x1,y1,x2,y2);d=math.atan2(y2-y1,x2-x1);q=c.beginPath();q.moveTo(x2,y2);q.lineTo(x2-8*math.cos(d-.4),y2-8*math.sin(d-.4));q.lineTo(x2-8*math.cos(d+.4),y2-8*math.sin(d+.4));q.close();c.drawPath(q,fill=1,stroke=0)
def finish():c.showPage()
start('試作スタンド 図解組立ガイド','モーター・胴体ケースなし / 市販ねじ・金属ナットで組み立てる固定式投影試作')
img('overview.png',80,297,435,400)
y=p('組み立てる順番',283,16);y=p('①部品確認 → ②台座 → ③支柱・架台 → ④プロジェクタ受け → ⑤大小の鏡 → ⑥頭部とアーム → ⑦顔・投影確認',y,12)
y=p('このガイドの対象',y-5,14);y=p('改訂07の鏡と支柱・改訂06の上下分割した後頭部を含む固定式試作です。本番の可動機構用の短い支柱とは混在させません。図は組立位置の説明用で、実寸の型紙ではありません。',y)
p('図では見やすさのため一部のねじを省略しています。実物の保持強度・放熱・投影品質は未試験です。',y,10)
finish()
start('01 / 部品と金具を揃える','STLはmm単位・100%倍率。組立位置のassembly/と、印刷用のprint/を区別します。')
y=table([['印刷部品','数','ファイル名の目印'],['台座2枚・連結板2枚・脚4個','8','base-left/right / base-join-1/2 / base-foot-1…4'],['頭部用支柱2本・上部アーム2個','4','head-column-left/right / head-arm-left/right'],['プロジェクタ架台4本・受け1個・クリップ4個','9','projector-stand-1…4 / drop-projector-tray-t5 / …-clip-0…3'],['小鏡架台・固定板・裏板','5','lower-mirror-stand / printed-mirror-* / drop-printed-foldMirror-*'],['ナット抜け止めカバー','4','tray-nut-cover-0…3'],['後頭部上下・顔・位置決めピン2本','5','head-upper / head-lower / face / head-alignment-pin-0…1']],690,[230,30,247])
y=table([['金具','数量','使用箇所'],['M4×25 / M4ナット / M4ワッシャー','26 / 30 / 56','台座等＋後頭部。ナット・座金は全M4分。座金厚0.5 mm'],['M3×20','4','プロジェクタの押さえクリップ'],['M3×12','4','プロジェクタ受けと架台'],['M3×12','10','鏡の締結6本＋大小の鏡架台4本'],['M4×20','4','後頭部上下の接続'],['M3×8','4','後頭部と上部アーム'],['M3ナット / M3ワッシャー','22 / 30','ナット対辺5.5、厚さ2.4 mm。ワッシャー厚0.5 mm想定']],y,[215,92,200])
p('別途：磁石Ø5.9×2.9 mmを8個、大小の前面反射鏡、鏡・磁石用の接着材、押さえ用パッド、脚のゴム底。工具は対応する六角レンチ、ナットを保持する工具、ノギスを用意します。鏡型2点と磁石・ナット確認用パーツ各1点は組立部品35点に含みません。',y,10)
finish()
start('02 / 台座を組み立てる','使う金具：M4×25を8本、M4ナット8個、M4ワッシャー16枚')
img('base.png',44,385,507,315)
y=p('図：台座を下から見た分解図。連結板と脚を下へ離して表示しています。',373,10)
y=p('1. base-left と base-right を合わせます。裏側の base-join-1/2 を使い、連結板1枚につきM4×25を2本通します。',y,12)
y=p('2. base-foot-1…4 を四隅へ各1本で固定します。脚下面のくぼみに、ねじ頭とワッシャーを収めます。締結後に脚のゴム底を付けます。',y,12)
y=p('3. ナット側にもワッシャーを入れ、ナットを工具で押さえて締めます。台座が平らで、ねじ頭が接地しないことを確認します。',y,12)
for x,w,t in [(44,65,'ねじ頭'),(121,60,'座金'),(193,130,'脚／連結板＋台座'),(335,65,'座金'),(412,125,'金属ナット')]:box(x,95,w,42,t)
for x1,x2 in [(109,121),(181,193),(323,335),(400,412)]:arrow(x1,116,x2,116)
p('締結順の模式図：丸穴は通すだけ。締まるねじ山は金属ナット側にあります。',83,9)
finish()
start('03 / 支柱と機器架台を立てる','使う金具：M4×25を14本、M4ナット14個、M4ワッシャー28枚')
img('frame.png',44,413,507,280)
y=p('1. head-column-left/right を各2本で固定。2. projector-stand-1…4 を各2本。3. lower-mirror-stand を2本で固定します。上部アームはまだ付けません。',402,11)
# Plan of actual base mounting datums. Not a scale template.
d=json.load(open(a/'generated-parts.json'));bh=d['baseHoles'];origin=(145,182);scale=.68
c.setStrokeColor(teal);c.setFillColor(gray);c.rect(origin[0]-120*scale,origin[1]-110*scale,240*scale,220*scale,fill=1,stroke=1);c.line(origin[0],origin[1]-110*scale,origin[0],origin[1]+110*scale)
labels=['頭L','頭R','P1','P2','P3','P4','小鏡']
for j,l in enumerate(labels):
 xx=(bh[2*j][0]+bh[2*j+1][0])/2;yy=(bh[2*j][1]+bh[2*j+1][1])/2;x=origin[0]+xx*scale;y=origin[1]+yy*scale;c.setFillColor(teal);c.circle(x,y,4,fill=1,stroke=0);p(l,y+(7 if j==1 else 20),9,x+(9 if j==1 else -14),45)
arrow(245,150,285,150);p('顔の正面方向',135,9,235,110)
p('上から見た位置関係',289,12,323,215);p('P1〜P4は projector-stand の末尾番号です。高さや向きが異なるため入れ替えません。<br/><br/>頭L／頭Rはファイル名の left／right を示します。見ている方向によって図の左右とは一致しません。<br/><br/>図は配置確認用です。穴あけ型紙には使いません。',265,10,323,215)
p('確認：架台の穴と台座の既存穴を合わせます。無理に曲げたり、追加で穴を開けたりしないでください。',84,10)
finish()
start('04 / プロジェクタを受けに固定','架台への固定：M3×12を4本 / 押さえ：M3×20を4本 / ナット8個・座金8枚')
img('projector.png',44,362,507,340)
y=p('図：プロジェクタを上へ離して表示。固定には底面の長方形部分を使います。',351,10)
y=p('1. drop-projector-tray-t5 の上側4か所へM3ナットを入れ、支持シュー4枚を横から差してロックします（9ページ参照）。架台側からM3×12とワッシャーを通し、受けを固定します。',y,12)
y=p('2. 実機の135×75 mmの底面を受けへ合わせます。モデルの受け開口は135.8×75.8 mmです。底のねじ位置は使わず、本体を加工しません。',y,12)
y=p('3. 4個のクリップに柔らかいパッドを当て、M3×20・ワッシャー・ナットで保持します。押さえるのは底面の縁だけ。基板・電子部品・レンズに当てないでください。',y,12)
y=p('4. HDMIと電源ケーブルを差せること、通風窓を塞いでいないことを確認します。ケーブルの重みは台座側で受け、端子へ力がかからないようにします。',y,12)
p('実物確認：底板厚さやパッドの圧縮量は未確定です。入らない・傾く場合は締め付けで解決せず、当たり箇所を確認してください。',y,10)
finish()
start('05 / 大小の鏡を組み立てる','締結用M3×12：各3本 / 架台用M3×12：各2本 / 各鏡でナット5個・座金5枚')
c.drawImage(str(a/'mirror-guide/large-exploded.png'),44,387,250,305,preserveAspectRatio=True,anchor='c')
c.drawImage(str(a/'mirror-guide/small-exploded.png'),305,387,246,305,preserveAspectRatio=True,anchor='c')
y=p('大鏡：直線10辺 ／ 小鏡：長方形。スリーブを使わない固定角度の機構です。',377,10)
y=p('1. 固定板の中央2個のナットを鏡側から、裏板の3個のナットを固定板側から入れます。鏡を接着する前に行います。',y,11)
y=p('2. 固定板を架台へM3×12と座金2組で取り付けます。裏板を高さ2.6 mmの一体型受け3か所へ密着させ、M3×12と座金3組で締結します。',y,11)
y=p('3. 鏡裏面に適合する接着材で3点接着します。設計厚1 mm。大鏡は直径12 mm相当、小鏡は直径6 mm相当の3点です。指定時間硬化させます。',y,11)
y=p('4. 大鏡は上側後頭部単体へ、小鏡は lower-mirror-stand へ取り付けます。角度調整のために締結ねじを緩めたまま使わないでください。',y,11)
p('大鏡125.2×158.4 mmの十角形、小鏡23.83×35.19 mm。大小とも鏡厚1.5 mm。詳細は「大小ミラー3D組立ガイド」とv7の実寸切り出し図を参照してください。',y,10)
finish()
start('06 / 後頭部下側と左右アームを接続','頭部側：M3×8を4本、M3ナット4個、座金12枚（1接続につき3枚）')
img('head.png',44,425,248,267)
c.drawImage(str(a.parent/'head-connectors-v3/neck-after.png'),307,449,244,200,preserveAspectRatio=True,anchor='c')
p('アームと後頭部の位置関係',417,9,53,240);p('ナットを入れる六角ポケット',417,9,308,240)
y=p('1. 首元内側の六角ポケットへM3ナットを左右2個ずつ入れます。磁石用の丸いポケットとは別です。',397,12)
y=p('2. head-arm-left/right の取付面でナットを覆い、アーム側からM3×8＋ワッシャー3枚で締めます。ワッシャー合計は1.5 mmです。',y,12)
for x,w,t in [(44,65,'M3×8'),(123,76,'座金3枚'),(213,73,'アーム'),(300,100,'ナット'),(414,123,'後頭部の受け')]:box(x,224,w,48,t,size=10)
for x1,x2 in [(109,123),(199,213),(286,300),(400,414)]:arrow(x1,247,x2,247)
y=p('締結順の模式図。ナットは後頭部の六角ポケットの中に入ります。',210,9)
y=p('3. 頭部を支えながら、左右アームの下端を支柱上端へM4×25で各2本固定します。ここはM4ナット4個・ワッシャー8枚を使います。落下しないよう補助者と作業してください。',y,12)
y=p('4. 大鏡は上側の後頭部へM3×12＋ワッシャー2組で固定します。上下の後頭部はM4×20を4本で接続します（10ページ）。',y,12)
p('M3×8の先端逃げは設計0.2 mmです。長いねじへ置き換えず、底付き前に取付面が接することを確認します。頭部保持・鏡取付台の接着は不要です。',y,10)
finish()
start('07 / 顔を取り付け、投影を確認','磁石：直径5.9×厚さ2.9 mmを8個 / 顔側4個・後頭部側4個')
img('face.png',44,423,507,274)
y=p('図：顔を離した状態。磁石は左右2組ずつ、合計4組です。',411,10)
y=p('1. 穴径確認用パーツで磁石の入り具合を確認します。各組の極性を確かめ、顔側・後頭部側を間違えないよう目印を付けます。',y,11)
y=p('2. 適合する二液エポキシをポケットの接着剤用溝に使い、磁石を着座させます。顔と後頭部は離した状態で硬化。合わせ面へ接着剤をはみ出させません。',y,11)
y=p('3. 硬化後、顔を手で支えながら合わせます。金具・ケーブルの噛み込み、合わせ面の隙間、磁石の脱落がないか確認します。',y,11)
y=p('通電前チェック',y-2,14)
for t in ['台座ががたつかず、すべてのナットが保持されている。','鏡・頭部・プロジェクタに緩みがなく、ねじ先が部品へ当たっていない。','レンズ・通風窓を塞がず、HDMI／電源端子にケーブルの荷重がかかっていない。']:
 y=p('□ '+t,y,10)
y=p('最後に白画像・格子画像で照射範囲とピントを確認し、鏡が所定位置に固定されていることを確認します。顔の端の影・ぼけ・二重像・光量は実機で確認します。設計データのみで投影品質を保証したものではありません。',y,10)
p('対応データ：projection-prototype-print.zip / 組立位置の参照：projection-prototype.blend またはブラウザの固定式試作ページ。',y,9)
finish()
start('08 / 上入れ式ナットと支持シュー','改訂05：トレーとシュー4枚を再印刷。架台のねじ4本をM3×12へ変更。')
img('tray-pocket-detail.png',44,405,507,290)
y=p('分解図：青緑＝受け、銀＝ナット、橙＝抜け止め兼支持シュー。',391,10)
y=p('1. 確認用パーツとシュー1枚をPETGで先に印刷。付属のprint/のSTLをそのままの方向で使います。図の浮いた配置では印刷しません。',y,11)
y=p('2. 六角穴は上向き。穴の底は造形台から続く厚さ2.35 mmの床で支えています。ナット穴内のサポートを必要としない形状です。',y,11)
y=p('3. 上からナットを入れ、通風窓側から外周へシューを差します。左右の爪をロックし、戻らずナットが落ちないことを確認します。',y,11)
y=p('4. 架台側からM3×12＋ワッシャーで締結。プロジェクタをシューの上面へ載せ、従来のクリップ4個で固定します。',y,11)
p('旧トレー・旧カバーは混在不可。支持面の高さと投影位置は維持しています。造形公差、爪の保持力、締結強度・温度は実物試験前です。',y,10)
finish()
start('09 / 後頭部を上下に分けて組む','補強接合：M4×20を4本、M4ナット4個、外径9×厚さ0.5 mmの座金4枚。')
img('split-overview.png',44,440,280,250);img('joint-detail.png',330,440,221,250)
y=p('左：上側を離した分解図。右：厚い接合座・リブ・位置決めピン。',430,10)
y=p('1. 付属STLは上下とも分割面を下にした印刷方向です。接合面・穴のバリだけを除き、位置決めピン2本の入り具合を確認します。',y,11)
y=p('2. 下側の内側から、M4ナット4個を横の溝へ差し込みます。仮保持には工具、または側面だけ少量の適合エポキシを使用。座面やねじ山には入れません。',y,11)
y=p('3. 上側単体へ大鏡を組みます。取り外す場合は固定ねじ2本を外し、鏡の法線方向へ14 mm離し、顔側へ進めながら100 mmにつき10 mm下げて引き出します。鏡を手で支えて作業します。',y,11)
y=p('4. 下側へピン2本を入れ、上側を合わせます。上側の内側から座金＋M4×20を入れ、3 mm六角レンチで4本を交互に少しずつ締めます。',y,11)
y=p('接合座は上10 mm・下8 mm。両側リブと内壁補強へ荷重を分散します。ピンは位置合わせ用で、締結は金属ねじとナットが担います。',y,11)
p('外形と顔の合わせ面は維持。上下ともAD5M・A1に収まる寸法です。実印刷での嵌合、締結強度、衝撃・繰り返し荷重は未試験。2 kgは設計上の仮定で、保証耐荷重ではありません。',y,10)
finish();c.save();shutil.copy2(out,a/'guide/assembly-guide.pdf');print(out)
