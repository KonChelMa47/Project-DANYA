exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];centerX=-22.01616859436035;centerY=.3457450866699219
angles=[0,44,136,180,240,300]
r=STEPControl_Reader();r.ReadFile(str(out/'upper-retaining-plate.step'));r.TransferRoots();a=r.OneShape()
# Coordinates of original annulus are local to yaw datum.
sector=[(r*math.cos(math.radians(t)),r*math.sin(math.radians(t)))for r,ts in [(70,range(45,136,2)),(51.5,range(135,44,-2))]for t in ts];a=cut(a,poly(sector,-14,4))
for deg in [44,136]:
 x,y=54*math.cos(math.radians(deg)),54*math.sin(math.radians(deg));a=fuse(a,cyl(3.2,-13.7,3,x,y));a=cut(a,cyl(1.65,-14,4,x,y))
save('trial-upper-retainer',a,'fixed',[centerX,centerY],'6061-T6 geometry candidate','Outer relief for swept outboard pitch bearing. Capturing lipR46..51.5 retained. Two upper bolt positions changed to44/136deg withOD6 spacers. Strength and printable/stock-hardware conversion NOT qualified; do not print3mm plate as-is.')
r=STEPControl_Reader();r.ReadFile(str(out/'yaw-bearing-base.step'));r.TransferRoots();a=r.OneShape()
for deg in [44,136]:
 x,y=54*math.cos(math.radians(deg)),54*math.sin(math.radians(deg));a=fuse(a,cyl(6,-28,3,x,y));a=cut(a,cyl(1.65,-28.1,3.2,x,y))
save('trial-yaw-bearing-base',a,'fixed',[centerX,centerY],'6061-T6 geometry candidate','Two additional through holes match new spacer positions; former holes retained. Same bearing/gear datum. Load and fabrication qualification pending.')
for i,deg in [(2,44),(3,136)]:
 x,y=centerX+54*math.cos(math.radians(deg)),centerY+54*math.sin(math.radians(deg));a=cut(cyl(3,-25,11.3,x,y),cyl(1.65,-25.1,11.5,x,y));save('trial-retainer-spacer-'+str(i),a,'fixed',[0,0],'metal spacer OD6 ID3.3 L11.3 candidate','Moved clear of swept outboard bearing and retained pulley envelope. Supplier length/fit not yet selected.')
(out/'trial-retainer.json').write_text(json.dumps({'status':'swept bearing clearance candidate; qualification pending','anglesDeg':angles,'parts':parts},indent=2));f=out/'trial-structural-candidate.json';d=json.load(open(f));names=['upper-retaining-plate','yaw-bearing-base','retainer-spacer-2','retainer-spacer-3']+[p['name']for p in parts];d['parts']=[p for p in d['parts']if p['name']not in names]+parts;f.write_text(json.dumps(d,indent=2));print('VALID',len(parts))
