import json,math
from pathlib import Path
exec(open('work/buildable-bust-v6/build.py').read().split('# Horn bolts:')[0])
from OCP.gp import gp_Trsf,gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
out=Path('outputs/danya-simulator/assets/buildable-bust-v8');parts=json.loads(Path('outputs/danya-simulator/assets/buildable-bust-v7/parts.json').read_text())['parts'];dat=json.load(open('work/buildable-bust-v8/datums.json'));drive=json.load(open(out/'drive-layout.json'));layout=json.load(open('work/torso-flat/torso-v3-layout.json'));interfaces=[]
def box(x,y,z,dx,dy,dz):return BRepPrimAPI_MakeBox(gp_Pnt(x,y,z),dx,dy,dz).Shape()
def cy(r,p,d,L):return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p),gp_Dir(*d)),r,L).Shape()
def move(s,m):
 t=gp_Trsf();t.SetValues(m[0],m[4],m[8],m[12],m[1],m[5],m[9],m[13],m[2],m[6],m[10],m[14]);return BRepBuilderAPI_Transform(s,t,True).Shape()
def point(p,m):return [sum(m[j*4+i]*p[j]for j in range(3))+m[12+i]for i in range(3)]
def qm(q,p):
 x,y,z,w=q;return [1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w),0,2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w),0,2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y),0,*p,1]
def sv(n,a,parent,m,mat,notes):save(n,move(a,m),parent,[0,0],mat,notes)
# Split motor saddles grip the rigid case through replaceable pads; no case screws removed.
for d,parent in zip(drive['motors'],['fixed','yaw','pitch']):
 role=d['role'];m=qm(d['quaternion'],d['tip']);a=box(12,-43.5,-18.4,12,41.8,36.8);a=cut(a,box(11.9,-38.3,-12.9,12.2,36.7,25.8))
 for z in [-15.65,15.65]:a=cut(a,cy(1.25,[18,-43.6,z],[0,1,0],42))
 # Rear frame interface two M3 holes, tapped, on 7mm spacing along X.
 for x in [14.5,21.5]:a=cut(a,cy(1.25,[x,-43.6,0],[0,1,0],6))
 sv('motor-'+role+'-saddle',a,parent,m,'6061-T6 aluminium','Split case saddle. 0.5mm nominal side clearance for 0.5mm silicone pads. Front cap two M3x45; rear two M3 frame screws. Do not use PCB or servo horn as a body support.')
 a=box(12,-1.7,-18.4,12,3,36.8)
 for z in [-15.65,15.65]:a=cut(a,cy(1.65,[18,-1.8,z],[0,1,0],3.2))
 sv('motor-'+role+'-cap',a,parent,m,'6061-T6 aluminium','Removable clamp cap, two M3x45 screws into saddle. Confirm friction/temperature in proof test.')
 interfaces.append(dict(name='motor-'+role,parent=parent,point=point([18,-43.5,0],m),matrix=m,holes=[point([x,-43.5,0],m)for x in [14.5,21.5]],axis=[m[4],m[5],m[6]]))
# Reflective plane unchanged; three back pads support glass without covering its front.
for key in ['mirror','foldMirror']:
 o=layout[key];m=dat[key]['matrix'];w,h=o['width'],o['height'];outline=o.get('outline',[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
 a=poly([(x*(w-2)*(.82 if key=='mirror' else .88),y*(h-2)*(.82 if key=='mirror' else .88))for x,y in outline],-5.1,3)
 mounts=[[-.25*w,-.18*h],[.25*w,-.18*h],[0,.25*h]]
 for x,y in mounts:a=cut(a,cyl(1.25,-5.2,3.2,x,y))
 sv(key+'-backing',a,'roll',m,'6061-T6 aluminium','Glass 1.1mm design thickness; three 1mm compliant adhesive pads. M3 tapped blind adjustment attachment. No front edge masking. Verify adhesive on actual coating/back surface.')
 for j,(x,y)in enumerate(mounts):
  sv(key+'-bond-pad-'+str(j+1),cyl(3 if key=='foldMirror' else 6,-2.1,1,x,y),'roll',m,'silicone adhesive','1mm bondline spacer-controlled. Actual bond strength NOT tested.')
 # Fixed triangular adjustment plate, separate from mirror backing.
 rr=4 if key=='foldMirror' else 7
 a=poly(mounts,-11.1,3)
 for x,y in mounts:a=fuse(a,cyl(rr,-11.1,3,x,y))
 for x,y in mounts:a=cut(a,cyl(1.65,-11.2,3.2,x,y))
 sv(key+'-adjustment-plate',a,'roll',m,'6061-T6 aluminium','Three M3 pull bolts with compression elastomer sleeves and locking nuts; nominal 3mm gap. Adjustment is for calibration, lock after alignment.')
 interfaces.append(dict(name=key,parent='roll',point=point([0,0,-11.1],m),matrix=m,holes=[point([x,y,-11.1],m)for x,y in mounts],axis=[m[8],m[9],m[10]]))
# Projector base reference from TI assembly: X0..135,Y0..3.97,Z0..75; origin at lens rim.
m=dat['projector']['matrix'].copy();m[12:15]=point([-111.62,-13.97,-81.33583308323333],m)
a=box(2,-6,2,131,3,71)
# Open tray centre, retain border and two cross ribs; air flow and wiring remain accessible.
for x0,dx in [(9,43),(60,66)]:a=cut(a,box(x0,-6.1,9,dx,3.2,57))
for x in [12,123]:
 for z in [7,68]:a=cut(a,cy(1.65,[x,-6.1,z],[0,1,0],3.2))
sv('projector-open-tray',a,'roll',m,'6061-T6 aluminium','135x75mm base datum, not Tripo surface. 3mm ventilated tray; 3mm base stand-off. Four M3 corner clamp screws. Existing TI base retained; no PCB loading.')
for j,(x,z)in enumerate([(12,7),(12,68),(123,7),(123,68)]):
 sv('projector-base-pad-'+str(j+1),cy(3,[x,-3,z],[0,1,0],3),'roll',m,'aluminium spacer3','Rigid 3mm datum pad under actual metal base; no PCB contact. Upper soft pads provide retention preload.')
interfaces.append(dict(name='projector',parent='roll',point=point([67.5,-6,37.5],m),matrix=m,holes=[point([x,-6,z],m)for x,z in [(12,7),(12,68),(123,7),(123,68)]],axis=[m[4],m[5],m[6]]))
(out/'parts.json').write_text(json.dumps({'status':'v8 mount candidate; verification pending','parts':parts},indent=2));Path('work/buildable-bust-v8/interfaces.json').write_text(json.dumps(interfaces,indent=2));print('parts',len(parts))
