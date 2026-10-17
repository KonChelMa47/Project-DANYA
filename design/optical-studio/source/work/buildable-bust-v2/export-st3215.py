import json
from pathlib import Path
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
p=Path('work/buildable-bust-v2');r=STEPControl_Reader();r.ReadFile(str(p/'st3215-cad/ST3215.step'));r.TransferRoots();s=r.OneShape();b=Bnd_Box();BRepBndLib.Add_s(s,b);bounds=b.Get();BRepMesh_IncrementalMesh(s,.08,False,.2,True).Perform();w=StlAPI_Writer();w.ASCIIMode=False;w.Write(s,'outputs/danya-simulator/assets/buildable-bust-v2/st3215-official.stl');(p/'st3215-cad-audit.json').write_text(json.dumps({'source':'https://files.waveshare.com/upload/5/59/ST3215-3D.zip','CAD':'ST3215.step','boundsMM':bounds,'sizeMM':[bounds[i+3]-bounds[i]for i in range(3)],'scope':'Official reference assembly incl idler. Coordinate axes must be registered before placement; not a nominal cuboid.'},indent=2));print(bounds)
