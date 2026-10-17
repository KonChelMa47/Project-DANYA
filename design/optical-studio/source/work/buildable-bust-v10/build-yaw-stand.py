exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];drive=json.load(open(out.with_name('buildable-bust-v9')/'drive-layout.json'));d=drive['motors'][0];m=qm(d['quaternion'],d['tip'])
a=box(12,-48.6,-10,17,6,20)
for z in [-5,5]:
 a=cut(a,cy(1.7,[20.5,-48.7,z],[0,1,0],6.2));a=cut(a,cy(3.1,[20.5,-48.7,z],[0,1,0],2.1))
a=move(a,m);start=point([20.5,-50,0],m);feet=[[-60,-60,-148],[-40,-60,-148]]
a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*start),5).Shape())
for end in feet:
 delta=[end[i]-start[i]for i in range(3)];a=fuse(a,cy(5,start,delta,math.sqrt(sum(v*v for v in delta))));a=fuse(a,cyl(9,-153,6,*end[:2]));a=cut(a,cyl(2.2,-153.1,6.2,*end[:2]))
delta=[feet[1][i]-feet[0][i]for i in range(3)];a=fuse(a,cy(4,feet[0],delta,math.sqrt(sum(v*v for v in delta))))
save('yaw-motor-rear-stand',a,'fixed',[0,0],'printed PETG candidate','Rearwards two-leg stand. Shaft datum preserved. Two M3x8 rear attachment screws in2mm head counterbores, twoM4 floor bolts. Floor interface and load test pending.')
for i,z in enumerate([-5,5]):
 a=fuse(cy(1.5,[20.5,-46.6,z],[0,1,0],8),cy(2.75,[20.5,-49.6,z],[0,1,0],3));sv('yaw-rear-stand-bolt-'+str(i),a,'fixed',m,'purchased M3x8','No washer in dimensioned recessed head stack. TipY-38.6 leaves0.4mm from centralcase rear.')
(out/'yaw-rear-stand.json').write_text(json.dumps({'status':'candidate','parts':parts},indent=2))
