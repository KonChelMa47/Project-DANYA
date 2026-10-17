exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
def tube(a,b,r=5.6):
 v=[b[i]-a[i]for i in range(3)];return cy(r,a,v,math.sqrt(sum(x*x for x in v)))
def chain(pts,r=5.6):
 a=tube(pts[0],pts[1],r)
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),r).Shape())
 for i in range(2,len(pts)):a=fuse(a,tube(pts[i-1],pts[i],r))
 return a
row=json.load(open('work/buildable-bust-v10/pitch-route-request-result.json'))[0];assert row['status']=='FOUND';row['points'][0]=[-10,53,7];row['points'][1][2]=4;a=chain(row['points'])
centerX=-22.01616859436035;centerY=.3457450866699219
b=cy(13,[centerX,centerY+55,-4],[0,1,0],6)
for x in [-16,12]:b=fuse(b,box(centerX+x,centerY+41,5,4,20,4));b=fuse(b,box(centerX+x,centerY+55,-4,4,6,13))
b=fuse(b,box(centerX-16,centerY+55,3,32,6,6))
for x in [-12,12]:
 b=fuse(b,cy(3,[centerX+x,centerY+45,4],[0,1,0],6));b=fuse(b,box(centerX+x-2,centerY+45,4,4,6,5));b=cut(b,cy(1.65,[centerX+x,centerY+44.9,4],[0,1,0],6.2))
b=cut(b,cy(4.2,[centerX,centerY+54.9,-4],[0,1,0],6.2));b=cut(b,cy(8.025,[centerX,centerY+56,-4],[0,1,0],5.1))
for deg in [90,210,330]:
 t=math.radians(deg);b=cut(b,cy(.8,[centerX+10*math.cos(t),centerY+57,-4+10*math.sin(t)],[0,1,0],4.1))
a=fuse(a,b)
d=json.load(open(out/'drive-layout.json'))['motors'][1];m=qm(d['quaternion'],d['tip']);a=fuse(a,move(box(12,-48.6,-10,17,6,20),m))
a=cut(a,move(box(-5,-42.6,-18,50,50,36),m))
for z in [-5,5]:a=cut(a,move(cy(1.7,[20.5,-75,z],[0,1,0],32.5),m));a=cut(a,move(cy(3.2,[20.5,-75,z],[0,1,0],28.4),m))
for other in ['yaw-upper-leg-1','pitch-bearing-cap-1']:
 rr=STEPControl_Reader();rr.ReadFile(str(out/(other+'.step')));rr.TransferRoots();b=rr.OneShape();t=gp_Trsf();t.SetTranslation(gp_Vec(centerX,centerY,0));a=cut(a,BRepBuilderAPI_Transform(b,t,True).Shape())
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
e=TopExp_Explorer(a,TopAbs_SOLID)
while e.More():
 pr=GProp_GProps();BRepGProp.VolumeProperties_s(e.Current(),pr);print('COMPONENT',pr.Mass(),pr.CentreOfMass().Coord(),flush=True);e.Next()
save('trial-pitch-support',a,'yaw',[0,0],'printed PETG candidate','Integrated outboard bearing bridge and servo rear support, radius5.6mm route. Bearing datum preserved. Two M3x8 rear bolts; continuous head/tool passages. Printed bearing fits/creep and2kg proof load not qualified.')
for i,z in enumerate([-5,5]):
 a=fuse(cy(1.5,[20.5,-46.6,z],[0,1,0],8),cy(2.75,[20.5,-49.6,z],[0,1,0],3));sv('trial-pitch-frame-bolt-'+str(i),a,'yaw',m,'M3x8','Envelope; rear captive nuts retained.')
(out/'trial-pitch-support.json').write_text(json.dumps({'status':'clearance candidate','parts':parts},indent=2));p=out/'trial-structural-candidate.json';data=json.load(open(p));data['parts']=[r for r in data['parts']if r['name']not in ['pitch-outboard-bridge','frame-pitch-motor-support']and not r['name'].startswith('trial-pitch-')]+parts;p.write_text(json.dumps(data,indent=2));print('VALID',len(parts))
