exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.gp import gp_Ax1
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
m=dat['projector']['matrix'].copy();m[12:15]=point([-111.62,-13.97,-81.33583308323333],m);t=gp_Trsf();t.SetValues(m[0],m[4],m[8],m[12],m[1],m[5],m[9],m[13],m[2],m[6],m[10],m[14])
r=STEPControl_Reader();r.ReadFile(str(out.with_name('buildable-bust-v9')/'projector-tray-t5.step'));r.TransferRoots();a=BRepBuilderAPI_Transform(r.OneShape(),t.Inverted(),True).Shape()
for x,z,dx,dz in [(-4.4,-4.4,1,1),(139.4,-4.4,-1,1),(-4.4,79.4,1,-1),(139.4,79.4,-1,-1)]:
 pts=[(x-dx*.2,z-dz*.2),(x+dx*14.4,z-dz*.2),(x-dx*.2,z+dz*14.4)];tool=poly(pts,0,24);tr=gp_Trsf();tr.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(1,0,0)),math.pi/2);tool=BRepBuilderAPI_Transform(tool,tr,True).Shape();tr=gp_Trsf();tr.SetTranslation(gp_Vec(0,12,0));tool=BRepBuilderAPI_Transform(tool,tr,True).Shape();a=cut(a,tool)
sv('projector-tray-t5',a,'roll',m,'printed PETG candidate','v10:14.4mm outer corner chamfers remove unused low corners. Measured135x75 base and135.8x75.8 straight locating rails, four support pads and four clamps unchanged. No mounting pattern from model screws. Physical load/fit testing pending.')
(out/'chamfered-tray.json').write_text(json.dumps({'status':'candidate','parts':parts},indent=2))
p=out/'structural-candidate.json'
if p.exists():
 data=json.load(open(p));data['parts']=[parts[0] if row['name']=='projector-tray-t5' else row for row in data['parts']];p.write_text(json.dumps(data,indent=2))
