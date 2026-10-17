exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
r=Path('outputs/danya-simulator/assets/projection-prototype');out=r/'prototype-v7';parts=[]
s=json.load(open(r/'layout.json'));selection=json.load(open(out/'mirror-selection.json'));polygon=[[round(x,1),round(y,1)]for x,y in selection['best']['poly']]
# Keep optical origin and reflective plane unchanged; outline is allowed to be asymmetric around it.
s['mirror']['width']=126;s['mirror']['height']=164;s['mirror']['outline']=[[x/126,y/164]for x,y in polygon]
(out/'layout.json').write_text(json.dumps(s,indent=2))
matrices={}
for key,prefix,lift in [('mirror','printed-mirror-',8),('foldMirror','drop-printed-foldMirror-',2)]:
 m=dat[key]['matrix'].copy();m[14]+=lift;matrices[key]=m;w,h=layout[key]['width'],layout[key]['height'];pts=[[-.25*w,-.18*h],[.25*w,-.18*h],[0,.25*h]]
 a=poly(pts,-15.6,5)
 for x,y in pts:a=fuse(a,cyl(4 if key=='foldMirror'else 7,-15.6,5,x,y))
 if key=='foldMirror':a=fuse(a,box(-9,-5,-15.6,18,10,5))
 for x,y in pts:
  a=fuse(a,cyl(3,-10.61,2.61,x,y));a=cut(a,cyl(1.7,-15.7,8.3,x,y))
 for x,y in ([[-5,0],[5,0]]if key=='foldMirror'else[[-10,0],[10,0]]):
  a=cut(a,cyl(1.7,-15.7,5.2,x,y));t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));a=cut(a,BRepBuilderAPI_Transform(hexagon(5.7,-13.2,2.7),t,True).Shape())
 sv(prefix+'fixed-plate',a,'static',m,'printed PETG','Rigid integral2.6mm pillars replace elastic sleeves. Same nominal reflective plane. Fixed angle; three M3x12 clamp bolts. No elastic adjustment.')
 # Rear stack shifts0.4mm to preserve the reflective plane with1.5mm glass.
 outline=layout[key].get('outline',[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
 back=poly([(x*(w-2)*(.82 if key=='mirror'else .88),y*(h-2)*(.82 if key=='mirror'else .88))for x,y in outline],-8,5.5)
 for j,(x,y)in enumerate(pts):
  t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));back=cut(back,BRepBuilderAPI_Transform(hexagon(5.7,-8.1,2.7),t,True).Shape());back=cut(back,cyl(1.7,-8.1,5.7,x,y))
  nut=BRepBuilderAPI_Transform(cut(hexagon(5.5,-7.9,2.4),cyl(1.3,-8,2.6)),t,True).Shape();sv(prefix+'adjust-nut-'+str(j),nut,'static',m,'purchased M3 hex nut','Rear-loaded before mirror. Shifted0.4mm for1.5mm glass.')
  sv(prefix+'bond-pad-'+str(j),cyl(3 if key=='foldMirror'else 6,-2.5,1,x,y),'static',m,'mirror-compatible adhesive1mm','Adhesive thickness design1mm. Physical bond verification pending.')
 sv(prefix+'backing',back,'static',m,'printed PETG','Shifted0.4mm rearward for1.5mm glass. Rests on2.6mm rigid pillars, no elastic sleeves.')
 if key=='foldMirror':sv('foldMirror-glass',poly([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]],-1.5,1.5),'static',m,'purchased front surface glass mirror','Thickness1.5mm. Reflective plane unchanged.')
 if key=='mirror':sv('mirror-glass',poly(polygon,-1.5,1.5),'static',m,'purchased front surface glass mirror','10 straight edges. Glass thickness1.5mm supplied by user. Optical origin unchanged; do not center asymmetric outline relative to backing.')
a=poly(polygon,0,2)
# Small recessed arrow marks positive mirror-localY, not a glass cut.
a=cut(a,poly([(-1,-3),(1,-3),(1,1),(3,1),(0,5),(-3,1),(-1,1)],1.3,.8))
save('mirror-cutting-master',a,'template',[0,0],'printed template','Trace OUTSIDE perimeter at100% scale. Ten straight edges; engraved arrow is orientation only, not a glass hole.')
for sg in [-1,1]:
 tag='left'if sg<0 else'right';x=-90;y=sg*85
 # A continuous flat back at y-10. All lateral outlines begin on the print bed.
 a=fuse(box(x-24,y-10,-160,48,20,8),box(x-9,y-10,-153,18,19,179))
 a=fuse(a,box(x-24,y-10,18,48,22,8))
 for z in [-160,18]:
  for xx in [x-16,x+16]:
   rr=2.2;a=cut(a,cyl(rr,z-.1,8.2,xx,y));a=cut(a,poly([(xx-rr/math.sqrt(2),y+rr/math.sqrt(2)),(xx+rr/math.sqrt(2),y+rr/math.sqrt(2)),(xx,y+rr*math.sqrt(2))],z-.1,8.2))
 save('head-column-'+tag,a,'static',[0,0],'printed PETG','Flat continuous back y-10; print that side down. M4 holes have45degree self-supporting teardrop roofs. Same end Z and bolt centers asv6.')
(out/'parts.json').write_text(json.dumps({'parts':parts,'matrices':matrices,'mirrorPolygonMM':polygon,'mirrorBoundingSizeMM':[max(p[j]for p in polygon)-min(p[j]for p in polygon)for j in [0,1]],'glassThicknessMM':1.5,'rigidPillarHeightMM':2.6,'physicalTests':False},indent=2));print('BUILT',len(parts))
