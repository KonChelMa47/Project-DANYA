exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Ax1
import shutil
parts=[];old=json.load(open('outputs/danya-simulator/assets/buildable-bust-v8/drive-layout.json'));new=json.loads(json.dumps(old))
angles={'yaw':-45,'pitch':15,'roll':0}
for i,d in enumerate(new['motors']):
 d['quaternion']=next(r['q'] for r in json.load(open(out/'backplate-motor-search.json'))['rows'] if r['role']==d['role'] and r['deg']==angles[d['role']])
(out/'drive-layout.json').write_text(json.dumps(new,indent=2))
# All selected modules retain the same shaft and optical datums.
for p in json.load(open(out/'motor-backplates.json'))['parts']:
 name=p['name'];r=STEPControl_Reader();r.ReadFile(str(out/(name+'.step')));r.TransferRoots();a=r.OneShape()
 role=name.split('-')[1];i=['yaw','pitch','roll'].index(role)
 if angles[role]:
  d=old['motors'][i];m=qm(d['quaternion'],d['tip']);axis=[m[4],m[5],m[6]];t=gp_Trsf();t.SetRotation(gp_Ax1(gp_Pnt(*d['tip']),gp_Dir(*axis)),math.radians(angles[role]));a=BRepBuilderAPI_Transform(a,t,True).Shape()
 save('selected-'+name,a,p['parent'],[0,0],p['material'],p['notes'])
for p in json.load(open(out/'projector-tray-candidates.json'))['parts']:
 if p['name'].startswith('projector-tray-t5'):
  parts.append(p)
assembly=TopoDS_Compound();builder=BRep_Builder();builder.MakeCompound(assembly)
for p in parts:
 r=STEPControl_Reader();r.ReadFile(str(out/(p['name']+'.step')));r.TransferRoots();builder.Add(assembly,r.OneShape())
w=STEPControl_Writer();w.Transfer(assembly,STEPControl_AsIs);w.Write(str(out/'mount-modules-assembly.step'))
(out/'mount-modules.json').write_text(json.dumps({'status':'mounting modules only; support frame and physical qualification pending','parts':parts},indent=2));print('MODULE PARTS',len(parts))
