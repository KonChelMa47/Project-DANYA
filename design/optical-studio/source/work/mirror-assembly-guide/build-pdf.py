from pathlib import Path
import json,shutil
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
r=Path('outputs/danya-simulator/assets/projection-prototype/mirror-guide');out=Path('output/pdf/DANYA-mirror-assembly-guide.pdf');pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'));c=canvas.Canvas(str(out),pagesize=A4);W,H=A4;c.setTitle('DANYA 大小ミラー 3D組立ガイド');page=0;ink=colors.HexColor('#17373c');teal=colors.HexColor('#297d80');labels=json.load(open(r/'render-labels.json'))
def p(t,y,size=10,x=42,w=511):
 q=Paragraph(t,ParagraphStyle('jp',fontName='JP',fontSize=size,leading=size*1.5,textColor=ink,wordWrap='CJK'));_,h=q.wrap(w,H);assert y-h>38,(page,t[:30],y,h);q.drawOn(c,x,y-h);return y-h-8
def start(t,sub):
 global page;page+=1;c.setFillColor(teal);c.rect(0,H-10,W,10,fill=1,stroke=0);p('PROJECT DANYA / MIRROR ASSEMBLY / STATIC v7',H-26,9);p(t,H-58,22);p(sub,H-96,10);c.setFont('JP',8);c.setFillColor(colors.gray);c.drawString(42,24,f'2026-09-24 / 現行STLから作成した組立図 / {page} / 4')
def img(n,x,y,w,h):c.drawImage(str(r/(n+'.png')),x,y,w,h,mask='auto')
def table(rows,y,widths):
 style=ParagraphStyle('t',fontName='JP',fontSize=9,leading=13,textColor=ink,wordWrap='CJK');tb=Table([[Paragraph(str(v),style)for v in row]for row in rows],colWidths=widths);tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf2f2')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c3d5d5')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));_,h=tb.wrap(511,H);assert y-h>40;tb.drawOn(c,42,y-h);return y-h-10
names=['締結ねじ M3×12 ×3','ワッシャー ×3','固定板＋高さ2.6 mmの一体型受け（ナット2個）','裏板（締結用ナット3個を先に入れる）','鏡用接着材：裏面の3点','厚さ1.5 mmの鏡：反射面を外側へ向ける']
for tag,title,sub in [('large','01 / 大鏡を組み立てる','頭部の上側に取り付ける鏡。外周約125.2×158.4（直線10辺） mm。'),('small','02 / 小鏡を組み立てる','プロジェクタ側の折り返し鏡。外周約23.83×35.19 mm。')]:
 start(title,sub+' 金色は鏡の識別色です。')
 x,y,w,h=61,328,437,370;img(tag+'-exploded',x,y,w,h)
 # Leader numbers use camera-projected layer levels; layers are deliberately separated.
 prev=310
 for num in range(1,7):
  px,py=labels[tag+'-exploded'][str(num)];ay=y+py*h;cy=max(ay,prev+19);prev=cy;c.setStrokeColor(teal);c.line(x+px*w,ay,530,cy);c.setFillColor(teal);c.circle(539,cy,8,fill=1,stroke=0);c.setFillColor(colors.white);c.setFont('JP',9);c.drawCentredString(539,cy-3,str(num))
 p('3D分解図：間隔を広げて表示。実際は裏板を一体型の受け3か所へ密着させて締めます。',320,9)
 rows=[['番号','部品／順序']]+[[str(i+1),n]for i,n in enumerate(names)];yy=table(rows,291,[40,471])
 prefix='printed-mirror-'if tag=='large'else'drop-printed-foldMirror-'
 yy=p('印刷部品：'+prefix+'fixed-plate.stl ／ '+prefix+'backing.stl',yy,9)
 p('ナットを入れる面は3ページ、架台への固定は4ページへ。鏡・接着材・金属ねじは印刷部品ではありません。',yy,9)
 c.showPage()
start('03 / ナットはこの面から入れる','固定板の2個は鏡側から。裏板の3個は固定板側から。鏡を接着する前に入れます。')
for tag,title,yy in [('large','大鏡',470),('small','小鏡',220)]:
 p(title,yy+226,14)
 img(tag+'-fixed-nuts',42,yy,249,205);img(tag+'-backing-nuts',305,yy,249,205)
 p('固定板：中央のナット2個',yy-2,10,42,248);p('裏板：ナット3個（裏面側）',yy-2,10,305,248)
y=p('橙色の六角形がM3ナットです。図では挿入前の位置に離して表示しています。',182,11)
y=p('① 六角ポケットのバリを取り、ナットを奥の座面まで入れます。<br/>② 固定板は中央2個、裏板は3個。締結ねじ用の丸穴と取り違えません。<br/>③ 仮組みではナットを工具で保持します。樹脂へ無理に圧入しないでください。',y,11)
p('ナット：対辺5.5×厚さ2.4 mmを設計基準としています。印刷公差による実際の入り具合は確認が必要です。',y,9);c.showPage()
start('04 / 架台へ固定し、鏡を仕上げる','締結用3本と取付用2本は別です。取付ねじもM3×12、各ワッシャー1枚。')
img('large-mount-detail',42,463,249,211);img('small-mount-detail',305,463,249,211)
p('大鏡：後頭部上側の一体型取付部',450,10,42,249);p('小鏡：lower-mirror-stand の上端',450,10,305,249)
y=p('取付部の3D分解図。固定板を前に、ねじを後ろに離しています。見やすさのため外殻と支柱下部を省略した図で、部品を切る指示ではありません。',424,9)
y=p('組立順',y,14)
y=p('1. 固定板へナット2個を入れ、架台の裏から「M3×12 → ワッシャー → 取付部 → 固定板内のナット」の順で2本を締めます。大鏡は後頭部を上下に分けた状態で作業します。',y,11)
y=p('2. 裏板へナット3個を入れます。固定板の後ろから締結ねじ＋座金を通し、裏板を固定板と一体の硬い受け3か所へ載せ、裏板のナットへねじ込みます。鏡なしで仮組みすると確認しやすくなります。',y,11)
y=p('3. 鏡の裏面に適合する接着材で裏板へ3点接着し、指定時間硬化させます。設計上は接着厚1 mm、大鏡は直径12 mm相当、小鏡は直径6 mm相当の3点です。反射面へ接着しません。',y,11)
y=p('4. 白・格子画像で照射を確認し、鏡が設計位置で固定されていることを確認します。角度調整用のねじではないため、緩めたまま使用しません。ねじの底付き・鏡のたわみ・ガタを確認します。',y,11)
y=p('用意する物（大小合計）',y,12)
y=p('M3×12ねじ10本／M3ナット10個／M3座金10枚／前面反射鏡2枚／鏡裏面に適合する接着材。大小とも鏡厚1.5 mm。スリーブは使用しません。固定角度での試作です。',y,10)
p('保持強度・接着材の適合性・造形公差・実機での投影品質は実機確認前です。この資料は現在の試作機構の組立説明です。',y,9)
c.showPage();c.save();shutil.copy2(out,r/'mirror-assembly-guide.pdf');print(out)
