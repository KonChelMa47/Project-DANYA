from pathlib import Path
import json,shutil
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
r=Path('outputs/danya-simulator/assets/projection-prototype/prototype-v7');out=Path('output/pdf/DANYA-prototype-v7-guide.pdf');pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'));c=canvas.Canvas(str(out),pagesize=A4);W,H=A4;ink=colors.HexColor('#17373c');teal=colors.HexColor('#297d80')
def p(t,y,size=10,x=42,w=511):
 q=Paragraph(t,ParagraphStyle('j',fontName='JP',fontSize=size,leading=size*1.5,textColor=ink,wordWrap='CJK'));_,h=q.wrap(w,H);assert y-h>35,(t,y,h);q.drawOn(c,x,y-h);return y-h-8
def start(title,sub,n):
 c.setFillColor(teal);c.rect(0,H-10,W,10,fill=1,stroke=0);p('PROJECT DANYA / STATIC PROTOTYPE v7',H-26,9);p(title,H-58,22);p(sub,H-96,10);c.setFillColor(colors.gray);c.setFont('JP',8);c.drawString(42,22,f'2026-09-24 / {n} / 2')
start('支柱と鏡を印刷・加工しやすく','鏡厚1.5 mm / 直線10辺 / スリーブなしの固定受け / 大支柱はサポートOFF',1)
c.drawImage(str(r/'columns-print.png'),42,453,330,212,preserveAspectRatio=True,mask='auto');c.drawImage(str(r/'column-hole-detail.png'),381,467,172,165,preserveAspectRatio=True,mask='auto')
y=p('左右支柱は平らな裏面を下へ。右図：横穴の上側を45度の山形にしています。',440,10)
y=p('大支柱：付属STLを回転せずに印刷',y,15)
y=p('head-column-left/right は各48×186×22 mm。裏面を上下の取付部まで同じ面につなぎました。M4のねじ軸位置は従来どおりです。サポートOFF、0.2 mm積層、PETG・外周5本以上を試作の開始条件とします。',y,11)
y=p('鏡：ゴムを使わず、裏板を硬い受けへ密着',y,15)
y=p('大小とも固定板へ高さ2.6 mmの受け3か所を一体化。裏板をM3×12＋座金3組で固定します。角度をねじの緩め加減で調整する機構ではありません。鏡厚1.5 mmに合わせて裏板を0.4 mm後退させ、反射面の位置を維持しています。',y,11)
y=p('更新する印刷部品：7ファイル',y,15)
y=p('左右の head-column 2本／大小の fixed-plate 2枚／大小の backing 2枚／大鏡の mirror-cutting-master。旧裏板と新固定板は混在させません。トレー・顔・後頭部は今回変更していません。',y,11)
y=p('大鏡：150×180 mm以上の元板から直線で切る',y,15)
y=p('仕上がりは125.2×158.4 mmの十角形。次ページは反射面側から見た実寸図です。樹脂の切り出し型は厚さ2 mmで、外周だけを転写します。刻印矢印は方向の目印です。',y,11)
p('CADで光路167,698点と変更部品の干渉を確認。支柱の45度超下向き面は0 mm²。実印刷・強度・切断精度・投影品質の実物試験は未実施です。',y,9);c.showPage()
start('大鏡の実寸切り出し図','反射面から見た向き。裏面に転写する場合は左右反転。PDF印刷は「実際のサイズ」。',2)
pts=json.load(open(r/'parts.json'))['mirrorPolygonMM'];ox=(W-150*mm)/2;oy=170
c.setStrokeColor(colors.HexColor('#aab8bb'));c.setLineWidth(.5);c.rect(ox,oy,150*mm,180*mm,fill=0,stroke=1)
c.setStrokeColor(ink);c.setLineWidth(.7);path=c.beginPath()
for i,(x,y)in enumerate(pts):
 xx=ox+(x+75)*mm;yy=oy+(y+90)*mm
 if i==0:path.moveTo(xx,yy)
 else:path.lineTo(xx,yy)
 c.setFillColor(ink);c.setFont('JP',8);c.drawCentredString(ox+(75+x*.90)*mm,oy+(90+y*.90)*mm,str(i+1))
path.close();c.drawPath(path,fill=0,stroke=1)
c.setStrokeColor(teal);cx=ox+75*mm;cy=oy+90*mm;c.line(cx,cy-15,cx,cy+15);c.line(cx,cy+15,cx-4,cy+8);c.line(cx,cy+15,cx+4,cy+8);c.setFillColor(ink);c.setFont('JP',10);c.drawCentredString(cx,cy-32,'反射面側 / +Y');c.setFont('JP',9);c.drawCentredString(cx,oy-16,'灰色枠：元板150×180 mm ／ 黒線：125.2×158.4 mmの十角形')
p('頂点1→2→…→10→1を直線で結ぶ外周です。外周の向きを裏板へ合わせます。使用するガラスの切断可否は材質・表面処理に応じて確認してください。',137,9)
c.setStrokeColor(ink);c.setLineWidth(1);xx=ox;yy=60;c.line(xx,yy,xx+50*mm,yy);c.line(xx,yy-4,xx,yy+4);c.line(xx+50*mm,yy-4,xx+50*mm,yy+4);c.setFont('JP',9);c.drawString(xx+50*mm+10,yy-3,'50 mm：印刷後に実測')
c.showPage();c.save();shutil.copy2(out,r/'prototype-v7-guide.pdf');print(out)
