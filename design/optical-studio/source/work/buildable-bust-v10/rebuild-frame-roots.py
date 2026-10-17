exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
import shutil
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];meta={p['name']:p for p in json.load(open(out/'trial-structural-candidate.json'))['parts']}
def read(name,freeze=False):
 path=out/(name+'.step')
 if freeze:
  frozen=out/(name+'-before-root-split.step')
  if not frozen.exists():shutil.copy2(path,frozen)
  path=frozen
 r=STEPControl_Reader();r.ReadFile(str(path));r.TransferRoots();return r.OneShape()
def rootbox(x,y,z):return fuse(fuse(cyl(4,z,3,x,y),cyl(4,z,3,x,y+9)),box(x-4,y,z,8,9,3))
def washer(name,x,y,z):save(name,cut(cyl(3.5,z,.5,x,y),cyl(1.6,z-.1,.7,x,y)),'roll',[0,0],'M3 washer OD7 t0.5','Envelope')
def nut(name,x,y,z):
 a=cut(hexagon(5.5,z,2.4),cyl(1.3,z-.1,2.6));t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));save(name,BRepBuilderAPI_Transform(a,t,True).Shape(),'roll',[0,0],'M3 hex nut AF5.5 t2.4','Thread envelope, not printable.')
def bolt(name,x,y,z,d):save(name,fuse(cyl(1.5,z-16 if d<0 else z,16,x,y),cyl(2.75,z if d<0 else z-3,3,x,y)),'roll',[0,0],'M3x16 socket screw','Realistic stock length, thread omitted.')
def spacer(name,x,y,z):save(name,cut(cyl(3,z,2,x,y),cyl(1.65,z-.1,2.2,x,y)),'roll',[0,0],'metal spacer OD6 ID3.3 L2','Purchased spacer envelope.')
# Supports meet opposite carrier faces, rather than occupying the carrier volume.
a=read('trial-frame-projector-right',True);a=cut(a,box(-300,-300,37,600,600,300));a=fuse(a,rootbox(4,-38,34));a=cut(a,cyl(1.7,33.9,3.2,4,-38));save('trial-frame-projector-right',a,'roll',[0,0],'interface geometry candidate','Underside carrier attachment atZ37; screw moved9mm away from descending support. Retained6mm metal rods; fabrication conversion pending.')
a=read('trial-frame-optics-left',True);top=cut(a,box(-300,-300,-300,600,600,343));bottom=cut(a,box(-300,-300,37,600,600,300));top=fuse(top,rootbox(-48,-38,43));bottom=fuse(bottom,rootbox(-48,-38,34));top=cut(top,cyl(1.7,42.9,3.2,-48,-38));bottom=cut(bottom,cyl(1.7,33.9,3.2,-48,-38));save('trial-left-head-rail',top,'roll',[0,0],'interface geometry candidate','Upper rail separated from descending projector support, clamped through carrier by sharedM3x16. Head shoe locations retained. Fabrication conversion pending.');save('trial-frame-optics-left',bottom,'roll',[0,0],'interface geometry candidate','Underside projector/fold-mirror support; no volume shared with carrier. Shared through-bolt atX-48Y-20, clear of rods. Fabrication conversion pending.')
a=read('frame-optics-right',True);a=fuse(a,rootbox(-48,21,43));a=cut(a,cyl(1.7,42.9,3.2,-48,21));save('trial-right-head-rail',a,'roll',[0,0],'interface geometry candidate','Upper head rail through-fastened beside support, rather than through rod axis. Head shape and shoe locations retained. Fabrication conversion pending.')
# Extend the carrier's existing integral bosses to the offset through-bolt positions.
a=read('roll-carrier-v7',True);cx0,cy0=meta['roll-carrier-v7']['xy']
for x,y in [(-48,-29),(4,-29),(-48,21)]:
 a=fuse(a,box(x-cx0-5,y-cy0-(14 if y==-29 else 5),37,10,19,6));hy=y-9 if y==-29 else y;a=cut(a,cyl(1.7,36.9,6.2,x-cx0,hy-cy0))
save('roll-carrier-v7',a,'roll',[cx0,cy0],meta['roll-carrier-v7']['material'],'Extended bosses with three3.4mm clearance holes, purchased through-bolts/nuts. Original axis geometry preserved; material/load qualification pending.')
bolt('root-left-bolt',-48,-38,46.5,-1);washer('root-left-upper-washer',-48,-38,46);washer('root-left-lower-washer',-48,-38,33.5);nut('root-left-nut',-48,-38,31.1)
bolt('root-projector-bolt',4,-38,31.5,1);spacer('root-projector-spacer',4,-38,31.5);washer('root-projector-lower-washer',4,-38,33.5);washer('root-projector-upper-washer',4,-38,43);nut('root-projector-nut',4,-38,43.5)
bolt('root-head-right-bolt',-48,21,48.5,-1);spacer('root-head-right-spacer',-48,21,46.5);washer('root-head-right-upper-washer',-48,21,46);washer('root-head-right-lower-washer',-48,21,36.5);nut('root-head-right-nut',-48,21,34.1)
(out/'frame-root-interfaces.json').write_text(json.dumps({'status':'through-bolt interfaces candidate, load and remaining contact checks pending','parts':parts},indent=2));f=out/'trial-structural-candidate.json';d=json.load(open(f));names=[p['name']for p in parts]+['frame-optics-right'];d['parts']=[p for p in d['parts']if p['name']not in names]+parts;f.write_text(json.dumps(d,indent=2));print('VALID',len(parts))
