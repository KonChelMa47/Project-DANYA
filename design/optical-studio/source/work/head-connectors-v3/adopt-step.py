exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
import shutil
v=Path('outputs/danya-simulator/assets/head-connectors-v3');out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
for side in ['left','right']:
 name='trial-'+side+'-head-rail';old=out/(name+'.step');archive=v/'archive-motorized'/old.name
 if not archive.exists():shutil.copy2(old,archive)
 reader=STEPControl_Reader();reader.ReadFile(str(v/('motorized-head-arm-'+side+'.step')));reader.TransferRoots();a=move(reader.OneShape(),[1,0,0,0,0,1,0,0,0,0,1,0,0,0,-8,1]);save(name,a,'roll',[0,0],'printed PETG candidate','head-connectors-v3; captive M3 head attachment, original rootinterface. Strength not physically qualified.')
