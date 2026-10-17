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
pairs=[('revised-rear-chassis','revised-floor'),('revised-projector-support-left','roll-carrier-v7'),('revised-projector-support-right','roll-carrier-v7'),('revised-projector-support-left','drop-printed-foldMirror-fixed-plate'),('drop-pitch-support','yaw-upper-leg-1'),('drop-pitch-support','pitch-bearing-cap-1')]
for i in range(6):
 for name in ['floor-button-bolt-','floor-bolt-washer-','floor-captive-nut-']:pairs.append((name+str(i),'revised-floor'))
for i in range(4):
 pairs.append(('lift-carrier-spacer-'+str(i),'yaw-ring-96T-retained'));pairs.append(('lift-carrier-spacer-'+str(i),'lift-carrier-countersunk-bolt-'+str(i)))
rows=[]
for a,b in pairs:
 c=BRepAlgoAPI_Common(shape(a),shape(b)).Shape();props=GProp_GProps();BRepGProp.VolumeProperties_s(c,props);r={'a':a,'b':b,'intersectionVolumeMM3':props.Mass()};rows.append(r);print(json.dumps(r),flush=True)
(out/'revised-mating-volumes.json').write_text(json.dumps({'scope':'Exact STEP nominal neutral-position overlap, common8mm lift and liftExempt respected. Surface contact allowed. Thread envelopes excluded from listed checks. Not strength certification.','rows':rows},indent=2))
