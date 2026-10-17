exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
source=out/'drop-projector-tray-t5-before-underedge.step'
if not source.exists():
 import shutil
 shutil.copyfile(out/'drop-projector-tray-t5.step',source)
r=STEPControl_Reader();assert r.ReadFile(str(source))==1;r.TransferRoots();a=r.OneShape()
m=dat['projector']['matrix'].copy();m[12:15]=point([-111.62,-13.97,-81.33583308323333],m);m[14]-=6
# Remove only underside overhang, preserve measured base locating wall Y>=-3.
a=cut(a,move(box(40,-8.2,67.5,32,5.15,13),m))
save('revised-tray-relief-candidate',a,'roll',[0,0],'printed PETG candidate','Measured135x75 base socket preserved; local underside edge relief X40..72 Z67.5..80.5 Y-8.2..-3.05 avoids pitch backplate. Physical stiffness/thermal/fit testing pending.')
print('VALID',len(parts))
