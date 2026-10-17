exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');meta={r['name']:r for r in json.load(open(out/'revised-structural-candidate.json'))['parts']};cache={}
def shape(name):
 if name not in cache:
  r=STEPControl_Reader();assert r.ReadFile(str(out/(name+'.step')))==1;r.TransferRoots();a=r.OneShape();d=meta[name];x,y=d['xy'];z=8 if d['parent'] in ['yaw','pitch','roll'] and not d.get('liftExempt') and name!='yaw-ring-96T-retained' else 0;t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,z));cache[name]=BRepBuilderAPI_Transform(a,t,True).Shape()
 return cache[name]
pairs=[('revised-roll-inner','pitch-yoke-v7'),('revised-roll-outer','pitch-yoke-v7'),('revised-roll-outer','roll-idler-axle-1'),('revised-roll-outer','revised-selected-backmount-roll-plate'),('mirror-head-cradle','printed-mirror-fixed-plate'),('revised-projector-support-left','drop-projector-tray-t5'),('revised-projector-support-right','drop-projector-tray-t5'),('drop-pitch-support','selected-backmount-pitch-plate')]
rows=[]
for a,b in pairs:
 if not all((out/(n+".step")).exists() for n in [a,b]):
  r={"a":a,"b":b,"status":"missing STEP; mesh contact needs separate verification"};rows.append(r);print(json.dumps(r),flush=True);continue
 c=BRepAlgoAPI_Common(shape(a),shape(b)).Shape();props=GProp_GProps();BRepGProp.VolumeProperties_s(c,props);r={'a':a,'b':b,'intersectionVolumeMM3':props.Mass()};rows.append(r);print(json.dumps(r),flush=True)
(out/'revised-support-mating-volumes.json').write_text(json.dumps({'scope':'Exact STEP nominal neutral-position overlap, common8mm lift and liftExempt respected. Surface contact allowed. Thread envelopes excluded from listed checks. Not strength certification.','rows':rows},indent=2))
