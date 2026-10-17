# Base dimensions supplied by user. No mounting pattern copied from bottom screws.
exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v9');parts=[]
m=dat['projector']['matrix'].copy();m[12:15]=point([-111.62,-13.97,-81.33583308323333],m)
# Exact lens position retained; measured base 135x75, CAD base thickness3.97 provisional.
for th in [5]:
 tag='projector-tray-t'+str(th);bottom=-3-th
 a=box(-4.4,bottom,-4.4,143.8,th,83.8)
 for xx,ww in [(8,43),(61,66)]:a=cut(a,box(xx,bottom-.1,8,ww,th+.2,59))
 # Full socket rails locate measured rectangle with0.4mm clearance on each side.
 rim=cut(box(-4.4,-3,-4.4,143.8,6,83.8),box(-.4,-3.1,-.4,135.8,6.2,75.8));
 for xx,ww in [(-4.5,18.5),(60,31)]:rim=cut(rim,box(xx,-.3,74,ww,3.5,6))
 a=fuse(a,rim)
 for xx in [12,123]:
  for zz in [7,68]:a=fuse(a,cy(3,[xx,-3,zz],[0,1,0],3))
 clips=[(16,-4),(48,-4),(80,-4),(96,79)]
 for j,(x,z)in enumerate(clips):
  a=fuse(a,cy(4.5,[x,bottom,z],[0,1,0],th));a=fuse(a,cy(3.5,[x,-3,z],[0,1,0],7.5));a=cut(a,cy(1.7,[x,bottom-.1,z],[0,1,0],th+7.7))
  z0=-8 if z<0 else 72
  clip=cut(box(x-5,4.5,z0,10,4,11),cy(1.7,[x,4.4,z],[0,1,0],4.2));sv(tag+'-clip-'+str(j),clip,'roll',m,'printed PETG candidate','Remove four M3 nuts to lift retaining clips. Clip bears on base edge through silicone; no PCB or lens contact. Actual base thickness to be checked.')
  sv(tag+'-pad-'+str(j),box(x-5,3.97,0 if z<0 else 72,10,.53,3),'roll',m,'silicone1mm nominal','Assembled gap0.53mm is geometric target, not confirmed preload.')
  bolt=fuse(cy(1.5,[x,bottom,z],[0,1,0],20),cy(2.75,[x,bottom-3,z],[0,1,0],3));sv(tag+'-bolt-'+str(j),bolt,'roll',m,'purchased M3x20','Thread omitted. No projector drilling. Base thickness and washer/nut engagement must match stack.')
  washer=cut(cy(3.5,[x,8.5,z],[0,1,0],.5),cy(1.6,[x,8.4,z],[0,1,0],.7));sv(tag+'-washer-'+str(j),washer,'roll',m,'purchased M3 washer OD7x0.5','Hardware envelope.')
  nut=cut(hex_y(x,9,z,2.4,5.5),cy(1.3,[x,8.9,z],[0,1,0],2.6));sv(tag+'-nut-'+str(j),nut,'roll',m,'purchased M3 hex nut','Plain nut, removable threadlocker after test. Not nyloc dimensions.')
 # Four underside attachment bores separate from projector base, retained for frame design.
 for x in [12,123]:
  for z in [7,68]:
   # Blind captive nuts loaded from underside: solid top1mm below datum pad.
   a=cut(a,cy(1.7,[x,bottom-.1,z],[0,1,0],th-1.0));a=cut(a,hex_y(x,bottom-.1,z,min(2.6,th-1.0)))
 
 for j,(x,z) in enumerate([(12,7),(12,68),(123,7),(123,68)]):
  nut=cut(hex_y(x,bottom+.1,z,2.4,5.5),cy(1.3,[x,bottom,z],[0,1,0],2.6));sv(tag+'-frame-nut-'+str(j),nut,'roll',m,'purchased M3 hex nut','Loaded from underside; final support must retain nut and cover access opening. Frame joint incomplete.')
 sv(tag,a,'roll',m,'printed PETG candidate','135.8x75.8mm socket, measured base135x75. Support pads lift base3mm; two large ventilation windows. Four captive M3 frame nuts from underside. Rear model screw positions NOT used. Frame, heat and physical retention qualification pending.')
(out/'projector-tray-candidates.json').write_text(json.dumps({'status':'base receiver candidates, frame connection pending','measuredBaseMM':[135,75],'cadBaseThicknessProvisionalMM':3.97,'clearancePerSideMM':.4,'parts':parts},indent=2));print('VALID',len(parts))
