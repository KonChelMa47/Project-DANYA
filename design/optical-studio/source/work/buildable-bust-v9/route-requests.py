import json
from pathlib import Path
D=json.load(open('work/buildable-bust-v8/datums.json'))
def pt(p,m):return [sum(m[j*4+i]*p[j] for j in range(3))+m[12+i] for i in range(3)]
m=D['projector']['matrix'];q=[-111.62,-13.97,-81.33583308323333]
req=[]
for i,(x,z,start) in enumerate([(12,7,[-48,-29,44]),(12,68,[-48,-29,44]),(123,7,[4,-29,44]),(123,68,[4,-29,44])]):
 end=pt([x+q[0],-11+q[1],z+q[2]],m);req.append(dict(name='projector-'+str(i),parent='roll',start=start,end=end,bounds=[[-95,-115,-145],[65,50,52]],ignore=['roll-carrier-v7','projector-tray-t5','projector-tray-t5-frame-nut-'+str(i)],radius=5))
m=D['mirror']['matrix']
for sign in [-1,1]:
 req.append(dict(name='mirror-'+str(sign),parent='roll',start=[-48,-29 if sign<0 else 30,44],end=pt([sign*10,0,-18.6],m),bounds=[[-95,-75,35],[35,75,260]],ignore=['roll-carrier-v7','printed-mirror-fixed-plate','printed-mirror-frame-nut-0','printed-mirror-frame-nut-1'],radius=5))
Path('work/buildable-bust-v9/printed-routes.json').write_text(json.dumps(req,indent=2))
p=Path('work/buildable-bust-v9/find-routes.mjs');s=p.read_text().replace('[],3.6)','[],(req.radius||5)+.6)').replace('radius:3','radius:req.radius||5');p.write_text(s)
