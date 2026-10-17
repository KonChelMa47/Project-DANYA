import json
from pathlib import Path
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
p=Path('work/buildable-bust-v2');r=STEPControl_Reader();r.ReadFile(str(p/'x430-official.step'));r.TransferRoots();s=r.OneShape();b=Bnd_Box();BRepBndLib.Add_s(s,b);bounds=b.Get();BRepMesh_IncrementalMesh(s,.08,False,.2,True).Perform();w=StlAPI_Writer();w.ASCIIMode=False;w.Write(s,'outputs/danya-simulator/assets/buildable-bust-v2/x430-official.stl');(p/'motor-cad-audit.json').write_text(json.dumps({'source':'https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/#drawings','CAD':'XM_H-430_idler.stp','boundsMM':bounds,'sizeMM':[bounds[i+3]-bounds[i]for i in range(3)],'scope':'Official reference assembly incl idler. Coordinate axes must be registered before placement; not a nominal cuboid.'},indent=2));print(bounds)
