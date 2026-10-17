# Printed split saddles; all retention by purchased M3 nuts, no tapped polymer.
exec(open('work/buildable-bust-v8/build-mounts.py').read().split('# Split motor saddles')[0])
from OCP.gp import gp_Ax1
out=Path('outputs/danya-simulator/assets/buildable-bust-v9');out.mkdir(exist_ok=True);parts=[]
def hex_y(x,y,z,h,af=5.7):
 a=hexagon(af,0,h);t=gp_Trsf();t.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(1,0,0)),-math.pi/2);a=BRepBuilderAPI_Transform(a,t,True).Shape();t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,z));return BRepBuilderAPI_Transform(a,t,True).Shape()
for d,parent in zip(drive['motors'],['fixed','yaw','pitch']):
 for xc in [18,22,26]:
  m=qm(d['quaternion'],d['tip']);tag=d['role']+'-'+str(xc)
  a=box(xc-8,-43.5,-20.2,16,41.8,40.4)
  a=cut(a,box(xc-8.1,-38.3,-12.9,16.2,36.7,25.8))
  for z in [-16.2,16.2]:a=cut(a,cy(1.7,[xc,-43.6,z],[0,1,0],42))
  for x in [xc-3.5,xc+3.5]:
   a=cut(a,cy(1.7,[x,-43.6,0],[0,1,0],6));a=cut(a,hex_y(x,-41.3,0,3.1))
  sv('printed-motor-'+tag+'-saddle',a,parent,m,'PETG candidate','16mm wide split saddle. Through M3x50 clamp bolts; two M3 hex nuts loaded from motor cavity before fitting motor for rear attachment. Rear nut pockets AF5.7/depth3.0; inspect print fit. No tapped plastic. Case clearance0.5mm nominal, pad/preload and temperature validation pending.')
  a=box(xc-8,-1.7,-20.2,16,3,40.4)
  for z in [-16.2,16.2]:a=cut(a,cy(1.7,[xc,-1.8,z],[0,1,0],3.2))
  sv('printed-motor-'+tag+'-cap',a,parent,m,'PETG candidate','Separate cap with two M3x50 through bolts, washers both sides and hex nuts. Printed material/fastener preload must be validated.')
  for j,z in enumerate([-16.2,16.2]):
   # M3x50 under-head atY1.8, washer0.5 oncapY1.3.
   a=fuse(cy(1.5,[xc,1.8,z],[0,-1,0],50),cy(2.75,[xc,1.8,z],[0,1,0],3))
   sv('hardware-'+tag+'-clamp-bolt-'+str(j),a,parent,m,'purchased M3x50 socket screw','Envelope; thread omitted. Not a printable part.')
   for k,y in enumerate([1.3,-44.0]):
    a=cut(cy(3.5,[xc,y,z],[0,1,0],.5),cy(1.6,[xc,y-.1,z],[0,1,0],.7));sv('hardware-'+tag+'-washer-'+str(j)+'-'+str(k),a,parent,m,'purchased M3 washer OD7 thickness0.5','Envelope; confirm supplier dimensions before final assembly.')
   a=cut(hex_y(xc,-46.4,z,2.4,5.5),cy(1.3,[xc,-46.5,z],[0,1,0],2.6));sv('hardware-'+tag+'-clamp-nut-'+str(j),a,parent,m,'purchased M3 hex nut','Thread omitted. Rear wrench access required; use removable threadlocker after validation.')
  for j,x in enumerate([xc-3.5,xc+3.5]):
   a=cut(hex_y(x,-41.2,0,2.4,5.5),cy(1.3,[x,-41.3,0],[0,1,0],2.6));sv('hardware-'+tag+'-rear-nut-'+str(j),a,parent,m,'purchased M3 hex nut','Insert from motor cavity before installing servo. Rear support and bolt length must match final frame.')
(out/'motor-candidates.json').write_text(json.dumps({'status':'isolated printable motor clamp candidates; not a complete assembly','parts':parts},indent=2))
print('valid CAD parts',len(parts))
