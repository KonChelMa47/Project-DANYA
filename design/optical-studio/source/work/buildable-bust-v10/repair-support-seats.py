exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
def read(n):
 r=STEPControl_Reader();assert r.ReadFile(str(out/(n+'.step')))==1;r.TransferRoots();return r.OneShape()
a=read('revised-roll-outer')
for y,z in [(60.34574508666992,-10),(83.67757844924927,-30)]:a=cut(a,cy(2.65,[0,y,z],[1,0,0],30))
save('revised-roll-outer-seatfix',a,'pitch',[0,0],'metal6mm candidate','Axle holes recut after motor-branch union. Print/stock fabrication pending.')
m=dat['projector']['matrix'].copy();m[12:15]=point([-111.62,-13.97,-81.33583308323333],m);m[14]-=6
for side,z in [('right',7),('left',68)]:
 a=read('revised-projector-support-'+side);a=cut(a,read('drop-projector-tray-t5'))
 from OCP.GProp import GProp_GProps
 from OCP.BRepGProp import BRepGProp
 from OCP.Bnd import Bnd_Box
 from OCP.BRepBndLib import BRepBndLib
 tr=gp_Trsf();tr.SetValues(m[0],m[4],m[8],m[12],m[1],m[5],m[9],m[13],m[2],m[6],m[10],m[14]);inv=tr.Inverted()
 e=TopExp_Explorer(a,TopAbs_SOLID);pieces=[]
 while e.More():
  pp=GProp_GProps();BRepGProp.VolumeProperties_s(e.Current(),pp);bb=Bnd_Box();BRepBndLib.Add_s(BRepBuilderAPI_Transform(e.Current(),inv,True).Shape(),bb);pieces.append({"v":pp.Mass(),"localBBox":bb.Get()});e.Next()
 print(side,'PIECES',pieces,flush=True)
 save('revised-projector-support-'+side+'-seatfix',a,'roll',[0,0],'metal6mm candidate','Receiver mating seat clipped to underside Y-8; base locating rim unchanged. Fabrication/strength pending.')
print('VALID',len(parts))
