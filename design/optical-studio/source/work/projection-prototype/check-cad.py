exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
root=Path('outputs/danya-simulator/assets/projection-prototype');pairs=json.load(open(root/'static-check.json'))['surfaceContacts'];cache={};rows=[]
for a,b in pairs:
 try:
  shapes=[]
  for n in [a,b]:
   if n not in cache:
    p=root/'assembly'/(n+'.step');lift=False
    if not p.exists():p=Path('outputs/danya-simulator/assets/buildable-bust-v10')/(n+'.step');lift=True
    if not p.exists():raise FileNotFoundError(n)
    r=STEPControl_Reader();assert r.ReadFile(str(p))==1;r.TransferRoots();s=r.OneShape()
    if lift:s=move(s,[1,0,0,0,0,1,0,0,0,0,1,0,0,0,8,1])
    cache[n]=s
   shapes.append(cache[n])
  common=BRepAlgoAPI_Common(*shapes).Shape();g=GProp_GProps();BRepGProp.VolumeProperties_s(common,g);rows.append({'pair':[a,b],'overlapMM3':g.Mass()})
 except FileNotFoundError:rows.append({'pair':[a,b],'check':'mesh only'})
(root/'mating-volumes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
