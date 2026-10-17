exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.gp import gp_Ax1
parts=[];centerY=g['center'][1]
def mirror(s):
 t=gp_Trsf();t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),math.pi);return BRepBuilderAPI_Transform(s,t,True).Shape()
def cyly(r,y,L,z=-4,x=0):return cy(r,[x,y,z],[0,1,0],L)
# Lower load plate grows downward; bearing datum remainsZ-25.
a=ring(46,36,-33,8)
for i in range(6):
 t=i*math.pi/3;x,y=50*math.cos(t),50*math.sin(t);a=fuse(a,cyl(8,-33,8,x,y))
a=cut(a,cyl(15,-34,10,dx,dy))
for i in range(6):
 t=i*math.pi/3;x,y=54*math.cos(t),54*math.sin(t);a=cut(a,cyl(1.7,-33.1,8.2,x,y));a=cut(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,-33),gp_Dir(0,0,1)),3.35,1.7,1.65).Shape())
save('printed-yaw-bearing-base',a,'fixed',[cx,centerY],'printed PETG candidate','8mm load plate, original bearing datum retained. Six M3x35 stack fasteners proposed; fixed printed frame attachment pending. Mechanical strength not yet qualified.')
a=fuse(ring(58,46,-13.7,3),ring(58,50,-10.7,5));a=cut(a,cyl(8,-14,10,dx,dy))
for i in range(6):
 t=i*math.pi/3;a=cut(a,cyl(1.7,-13.8,8.2,54*math.cos(t),54*math.sin(t)))
save('printed-yaw-retainer',a,'fixed',[cx,centerY],'printed PETG candidate','Stepped rim:3mm inner thrust lip,8mm outer flange. Retainer bolt stack must be rechecked. No machined aluminium.')
# Printed yaw-stage bearing pedestals. Metal inserts replace tapped plastic.
for sy in [-1,1]:
 a=fuse(box(-17,39,-8.5,34,5.5,6),cyly(14.8,39,6));a=cut(a,cyly(4.2,38.9,6.2));a=cut(a,cyly(8.1,40,5.1))
 for x in [-12,12]:a=fuse(a,box(x-4,37,-8.5,8,8,6))
 for x in [-12,12]:
  a=cut(a,cyl(2,-8.6,5.1,x,41))
  s=cut(cyl(2.3,-8.5,4,x,41),cyl(1.3,-8.6,4.2,x,41));save('insert-leg-'+str(sy)+'-'+str(x),s if sy>0 else mirror(s),'yaw',[cx,centerY],'ruthex RX-M3Sx4.0 candidate','4mm insert,4mm installation hole with5mm depth; nominal envelope includes knurl interference. Fit coupon before assembly.')
 for j,deg in enumerate([90,210,330]):
  t=math.radians(deg);x,z=11.5*math.cos(t),-4+11.5*math.sin(t);a=cut(a,cyly(1.6,40,5.1,z,x))
  s=cut(cyly(1.8,41,4,z,x),cyly(.85,40.9,4.2,z,x));save('insert-bearing-'+str(sy)+'-'+str(j),s if sy>0 else mirror(s),'yaw',[cx,centerY],'ruthex RX-M2x4','Official nominal envelope3.6x4; installation hole3.2,depth5. Not a printable part.')
 a=cut(a,box(-40,30,-40,80,30,26))
 if sy>0:
  # Dedicated motor support boss, separate from bearing seat.
  a=fuse(a,box(12,39,-8.5,7,6,6));a=fuse(a,cyly(5,39,6,-4,18));a=cut(a,cyly(2,40,5.1,-4,18))
  s=cut(cyly(2.3,41,4,-4,18),cyly(1.3,40.9,4.2,-4,18));save('insert-pitch-frame',s,'yaw',[cx,centerY],'ruthex RX-M3Sx4.0 candidate','Dedicated pitch-motor support fixing, front insertion; no bearing pocket intrusion.')
 cap=cut(cyly(14.8,45,3),cyly(4.2,44.9,3.2));cap=cut(cap,box(-40,30,-40,80,30,26))
 for j,deg in enumerate([90,210,330]):
  t=math.radians(deg);x,z=11.5*math.cos(t),-4+11.5*math.sin(t);cap=cut(cap,cyly(1.1,44.9,3.2,z,x))
  b=fuse(cyly(1,42,6,z,x),cyly(1.75,48,2,z,x));save('bearing-cap-bolt-'+str(sy)+'-'+str(j),b if sy>0 else mirror(b),'yaw',[cx,centerY],'purchased M2x6 socket screw','3mm cap and3mm insert engagement. Thread geometry omitted.')
 save('printed-yaw-leg-'+str(sy),a if sy>0 else mirror(a),'yaw',[cx,centerY],'printed PETG candidate','Bearing seat16.2mm prototype fit. OD29.6 housing, PCD23 cap fasteners. M3 short inserts in feet. Actual fit and pullout tests pending.')
 save('printed-pitch-cap-'+str(sy),cap if sy>0 else mirror(cap),'yaw',[cx,centerY],'printed PETG candidate','3mm end cap with3xM2 screws. 8.4mm shaft throat. Print orientation/fit qualification pending.')
(out/'core-candidates.json').write_text(json.dumps({'status':'printed load-plate and bearing-pedestal candidates; not adopted until interference checks','parts':parts},indent=2));print('VALID',len(parts))
