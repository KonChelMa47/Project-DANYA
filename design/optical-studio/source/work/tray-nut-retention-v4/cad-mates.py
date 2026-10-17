exec(open('work/tray-nut-retention-v4/build.py').read().split("# Current v10")[0])
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
root=out.parent;rows=[]
def read(p):
 r=STEPControl_Reader();assert r.ReadFile(str(p))==1;r.TransferRoots();return r.OneShape()
def volume(a):
 p=GProp_GProps();BRepGProp.VolumeProperties_s(a,p);return abs(p.Mass())
tray=read(out/'drop-projector-tray-t5.step');cap=read(out/'tray-nut-cover-local.step')
for i,(x,z,sg)in enumerate([(12,7,1),(12,68,-1),(123,7,1),(123,68,-1)]):
 stand=read(root/'assembly'/('projector-stand-'+str(i+1)+'.step'));cv=read(out/('tray-nut-cover-'+str(i)+'.step'));rows.append({'joint':i,'trayStandOverlapMM3':volume(BRepAlgoAPI_Common(tray,stand).Shape()),'coverStandOverlapMM3':volume(BRepAlgoAPI_Common(cv,stand).Shape())})
 # T-rail sliding path with the two elastic hook noses removed. Nose interference is deliberate during snap insertion.
 core=cut(cap,box(5.5,-8.1,2,1,1.2,3));core=cut(core,box(-6.5,-8.1,2,1,1.2,3));path=[]
 for dz in range(15):
  M=[sg,0,0,0,0,1,0,0,0,0,sg,0,x,0,z+sg*dz,1];a=move(move(core,M),pm);path.append(volume(BRepAlgoAPI_Common(a,tray).Shape()))
 rows[-1]['maxNonDetentSlidingOverlapMM3']=max(path)
print(rows);(out/'mating-check.json').write_text(json.dumps(rows,indent=2));assert all(max(x[k]for k in ['trayStandOverlapMM3','coverStandOverlapMM3','maxNonDetentSlidingOverlapMM3'])<1e-5 for x in rows)
