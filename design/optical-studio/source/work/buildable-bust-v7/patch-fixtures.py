import json
from pathlib import Path
source=open('work/buildable-bust-v6/build.py').read();exec(source.split('# Horn bolts:')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v7');parts=json.loads((out/'parts.json').read_text())['parts'];parts=[p for p in parts if p['name'] not in ['test-interface-plate','test-load-lever']]
def box(x,y,z,dx,dy,dz):return BRepPrimAPI_MakeBox(gp_Pnt(x,y,z),dx,dy,dz).Shape()
s=open('work/buildable-bust-v7/build.py').read();exec(s[s.index('# Standalone'):])
