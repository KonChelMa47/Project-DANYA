exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');meta={r['name']:r for r in json.load(open(out/'trial-structural-candidate.json'))['parts']};cache={}
def shape(name):
 if name not in cache:
  r=STEPControl_Reader();assert r.ReadFile(str(out/(name+'.step')))==1;r.TransferRoots();a=r.OneShape();x,y=meta[name]['xy'];t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));cache[name]=BRepBuilderAPI_Transform(a,t,True).Shape()
 return cache[name]
rows=[]
pairs=[('trial-rear-chassis','yaw-bearing-base'),('trial-rear-chassis','trial-floor'),('trial-pitch-support','yaw-upper-leg-1'),('trial-pitch-support','pitch-bearing-cap-1'),('trial-frame-projector-right','roll-carrier-v7'),('trial-frame-optics-left','roll-carrier-v7'),('trial-left-head-rail','roll-carrier-v7'),('trial-right-head-rail','roll-carrier-v7')]
for name,other in pairs:
 c=BRepAlgoAPI_Common(shape(name),shape(other)).Shape();props=GProp_GProps();BRepGProp.VolumeProperties_s(c,props);r={'a':name,'b':other,'intersectionVolumeMM3':props.Mass()};rows.append(r);print(json.dumps(r),flush=True)
(out/'repaired-mating-volumes.json').write_text(json.dumps({'scope':'Exact STEP positive-volume overlap only; mounting surface contact is allowed. Latest outward roots, clipped rear/pitch supports. Not a strength test.','rows':rows},indent=2))
