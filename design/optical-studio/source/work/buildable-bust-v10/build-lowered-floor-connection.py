exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
def read(n):
 r=STEPControl_Reader();assert r.ReadFile(str(out/(n+'.step')))==1;r.TransferRoots();return r.OneShape()
def nutshape(x,y,z):
 a=cut(hexagon(7,z,3.2),cyl(1.7,z-.1,3.4));t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));return BRepBuilderAPI_Transform(a,t,True).Shape()
def pocket(x,y,z,h):
 t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));return BRepBuilderAPI_Transform(hexagon(7.2,z,h),t,True).Shape()
a=read('trial-rear-chassis')
for y in [-60,60]:
 a=fuse(a,cyl(9,-156.7,10.5,-55,y));a=cut(a,cyl(2.2,-156.8,20,-55,y));a=cut(a,pocket(-55,y,-149.6,20))
save('revised-rear-chassis',a,'fixed',[0,0],'printed PETG candidate','Feet extended4.5mm to meet lowered floor. CaptiveM4 hex nuts attop. Print/load/tool access not qualified.')
pts=[(-8+76.3*math.cos(i*math.pi/48),.345745+104.3*math.sin(i*math.pi/48))for i in range(96)];a=poly(pts,-163.5,6.8);a=cut(a,box(-37,-35,-163.6,60,70,7))
holes=[(-55,-60,16,-149.5),(-55,60,16,-149.5),(-75,0,12,-152.95),(20,-85,12,-152.95),(-10,-98,12,-152.95),(-10,98,12,-152.95),(20,85,12,-152.95)]
for i,(x,y,length,nz)in enumerate(holes):
 a=cut(a,cyl(2.2,-163.6,7,x,y));a=cut(a,cyl(4.75,-163.6,3.3,x,y))
 save('floor-button-bolt-'+str(i),fuse(cyl(2,-161.1,length,x,y),cyl(3.8,-163.3,2.2,x,y)),'fixed',[0,0],'M4x'+str(length)+' ISO7380 button screw envelope','HeadD7.6 H2.2; shaft/thread envelope, not printable. Counterbore seatZ-160.3; washer0.8.')
 save('floor-bolt-washer-'+str(i),cut(cyl(4.5,-161.1,.8,x,y),cyl(2.15,-161.2,1,x,y)),'fixed',[0,0],'M4 washer OD9 ID4.3 t0.8','Purchased envelope, supplier tolerance unverified.')
 save('floor-captive-nut-'+str(i),nutshape(x,y,nz),'fixed',[0,0],'M4 hex nut AF7 h3.2','Thread envelope. Rear feet/printed shell bosses have7.2mm pockets. Thread retention/load unqualified.')
save('revised-floor',a,'fixed',[0,0],'printed PETG candidate','152.6x208.6mm,6.8mm floor atZ-163.5..-156.7. Seven counterbored button screws. Captive nut bosses for shell must be fused and verified before assembly.')
(out/'lowered-floor-connection.json').write_text(json.dumps({'parts':parts,'status':'floor/rearfoot candidate; shellbosses pending'},indent=2));p=out/'revised-structural-candidate.json';d=json.load(open(p));names={p['name']for p in parts}|{'trial-rear-chassis','drop-floor'};d['parts']=[p for p in d['parts']if p['name']not in names]+parts;p.write_text(json.dumps(d,indent=2));print('VALID',len(parts))
