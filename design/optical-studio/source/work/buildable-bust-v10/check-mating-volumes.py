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
for row in json.load(open(out/'lift-full-assembly-screen.json'))['rows']:
 name=row['name']
 if not name.startswith(('trial-','frame-optics','mirror-head')) or name not in meta:continue
 for other in row['hits']:
  if other not in meta or meta[name]['parent']!=meta[other]['parent']or not (out/(other+'.step')).exists()or not(out/(name+'.step')).exists():continue
  a=shape(name);b=shape(other);c=BRepAlgoAPI_Common(a,b).Shape();props=GProp_GProps();BRepGProp.VolumeProperties_s(c,props);r={'a':name,'b':other,'intersectionVolumeMM3':props.Mass()};rows.append(r);print(json.dumps(r),flush=True)
(out/'mating-volume-check.json').write_text(json.dumps({'scope':'Same-parent STEP parts; initial overlap equals all poses for these rigid pairs. Thread envelopes may intentionally intersect; other positive volumes are defects, not automatically accepted mating. Current geometry, not archived triangle-screen meshes.','rows':rows},indent=2))
