exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
out=Path('outputs/danya-simulator/assets/projection-prototype/split-head-v6');parts=[];data=json.load(open(out/'head-split-check.json'));I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
for i,p in enumerate(data['flanges']):
 x,y,z=p['boltCenter'];sv(f'head-seam-bolt-{i}',fuse(cyl(2,155.5,20,x,y),cyl(3.5,175.5,4,x,y)),'static',I,'purchased M4x20 socket screw','Insert from upper shell.0.5mmwasher. Tip155.5, lower nut157.2. Physical stack check required.')
 sv(f'head-seam-washer-{i}',cut(cyl(4.5,175,.5,x,y),cyl(2.2,174.9,.7,x,y)),'static',I,'purchased M4 washer OD9 t0.5','Upper flange top')
 sv(f'head-seam-nut-{i}',move(cut(hexagon(7,157.2,3.2),cyl(1.75,157.1,3.4)),[1,0,0,0,0,1,0,0,0,0,1,0,x,y,0,1]),'static',I,'purchased M4 hex nut AF7 t3.2','Lower shell bottom-loaded nut well; shoulder160.5 transfers preload. Spot epoxy on nut side only for preassembly retention, not a structural adhesive.')
for p in data['pins']:
 x,y,z=p['center'];a=fuse(cyl(1.975,162.4,5.2,x,y),BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,162),gp_Dir(0,0,1)),1.575,1.975,.4).Shape());a=fuse(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,167.6),gp_Dir(0,0,1)),1.975,1.575,.4).Shape());sv(f'head-alignment-pin-{p["id"]}',a,'static',I,'printed PETG','Separate3.95mmx6mm pin,4.4mm blind sockets. Secondary upper socket elongated. Not load-bearing.')
(out/'hardware.json').write_text(json.dumps({'parts':parts},indent=2));print('PARTS',len(parts))
