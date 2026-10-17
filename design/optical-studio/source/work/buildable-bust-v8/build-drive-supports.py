exec(open('work/buildable-bust-v8/build-mounts.py').read().split('# Split motor saddles')[0])
from OCP.STEPControl import STEPControl_Reader
cy0=drive["pivot"][1]
parts=json.load(open(out/'parts.json'))['parts'];parts=[p for p in parts if not p['name'].startswith(('pitch-outboard','roll-idler','axis-shaft-pitch-1'))]
# Outboard 688 bearing supports the overhung pitch drive pulley.
a=cy(13,[cx,cy0+55,-4],[0,1,0],6)
a=cut(a,cy(4.2,[cx,cy0+54.9,-4],[0,1,0],6.2));a=cut(a,cy(8.025,[cx,cy0+56,-4],[0,1,0],5.1))
for x in [-16,12]:a=fuse(a,box(cx+x,cy0+41,11,4,20,5));a=fuse(a,box(cx+x,cy0+55,-4,4,6,20))
a=fuse(a,box(cx-16,cy0+55,10,32,6,6))
for x in [-12,12]:
 a=fuse(a,cy(3,[cx+x,cy0+45,4],[0,1,0],3));a=fuse(a,box(cx+x-2,cy0+45,4,4,3,12));a=cut(a,cy(1.65,[cx+x,cy0+44.9,4],[0,1,0],3.2))
for deg in [90,210,330]:
 t=math.radians(deg);a=cut(a,cy(.8,[cx+10*math.cos(t),cy0+57,-4+10*math.sin(t)],[0,1,0],4.1))
save('pitch-outboard-bridge',a,'yaw',[0,0],'6061-T6 aluminium','Additional 688 seat at Y+56..61; pulley remains Y+70. Two M3 screws into added yaw leg bosses at X+/-12 Z4. Housing paired with existing bearing: support span16mm; residual pulley overhang11.5mm.')
a=cut(cy(13,[cx,cy0+61,-4],[0,1,0],1.5),cy(4.2,[cx,cy0+60.9,-4],[0,1,0],1.7))
for deg in [90,210,330]:
 t=math.radians(deg);a=cut(a,cy(1.1,[cx+10*math.cos(t),cy0+60.9,-4+10*math.sin(t)],[0,1,0],1.7))
save('pitch-outboard-cap',a,'yaw',[0,0],'6061-T6 aluminium','Three M2x5 cap screws PCD20; 688 end retention.')
a=cut(cy(8,[cx,cy0+56,-4],[0,1,0],5),cy(4,[cx,cy0+55.9,-4],[0,1,0],5.2));save('pitch-outboard-bearing',a,'yaw',[0,0],'688ZZ purchased','8x16x5mm purchased bearing.')
a=cy(4,[0,34.5,-4],[0,1,0],40.5);a=cut(a,cy(1.5,[-5,36.5,-4],[1,0,0],10));save('axis-shaft-pitch-1',a,'pitch',[cx,cy0],'steel8h6','40.5mm shaft, Y34.5..75. Pulley clamping hub, cross-pin at36.5, M3 axial end retention. Load test still required.')
# Two-sided idler carrier and axles, shaft is supported at both ends.
trans=json.load(open(out/'transmission.json'));idler=[p for p in trans['routes'][1]['pulleys']if p['index']in [1,3]]
for side,x in [('inner',5),('outer',22.5)]:
 pts=([(36,10),(59,-3),(60.345745,-10),(94.103949,-24)] if side=='inner' else [(24,12),(60.345745,-10),(94.103949,-24)]);a=None
 for i in range(1,len(pts)):
  p0=[4.5 if side=="inner" and i==1 else x,pts[i-1][0],pts[i-1][1]];p1=[x,pts[i][0],pts[i][1]];v=[p1[k]-p0[k]for k in range(3)];L=math.sqrt(sum(q*q for q in v));rod=cy(3,p0,v,L);a=rod if a is None else fuse(a,rod)
 for y,z in [pts[0],pts[-2],pts[-1]]:a=fuse(a,cy(6,[(4.5 if side=="inner" and (y,z)==pts[0] else x)-1.5,y,z],[1,0,0],3))
 for p in idler:a=cut(a,cy(2.65,[x-6,p['center'][1],p['center'][2]],[1,0,0],12))
 a=fuse(a,box(3 if side=='inner' else 21,32 if side=='inner' else 20,4,3,8,13))
 for zz in [8,14]:a=cut(a,cy(1.65,[0,36 if side=='inner' else 24,zz],[1,0,0],28))
 rr=STEPControl_Reader();rr.ReadFile('outputs/danya-simulator/assets/buildable-bust-v7/pitch-yoke-v7.step');rr.TransferRoots();tt=gp_Trsf();tt.SetTranslation(gp_Vec(cx,cy0,0));a=cut(a,BRepBuilderAPI_Transform(rr.OneShape(),tt,True).Shape())
 save('roll-idler-carrier-'+side,a,'pitch',[0,0],'steel rod6 with machined end lugs','Two-sided M5 smooth-shoulder axles; 5mm rods and end bosses. Attachment to pitch yoke at Y24 Z12 requires matched bolt bosses. 3mm spacers outside pulley. MR85 bearing envelopes preliminary until load rating checked.')
for p in idler:
 y,z=p['center'][1:];j=str(p['index']);a=cy(2.5,[3,y,z],[1,0,0],23);save('roll-idler-axle-'+j,a,'pitch',[0,0],'steel shoulder axle M5','Smooth diameter5mm through bearing rings, end M5 thread + washer/nyloc. Do not run bearing on threads.')
(out/'parts.json').write_text(json.dumps({'status':'v8 support candidates; validation pending','parts':parts},indent=2))
