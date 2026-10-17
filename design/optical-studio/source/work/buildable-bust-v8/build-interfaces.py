exec(open('work/buildable-bust-v8/build-mounts.py').read().split('# Split motor saddles')[0])
from OCP.STEPControl import STEPControl_Reader
parts=[p for p in json.load(open(out/'parts.json'))['parts']if not p['name'].startswith('floor-spacer-')]
def load(n):
 p=next(p for p in parts if p['name']==n);r=STEPControl_Reader();r.ReadFile(str(out/(n+'.step')));r.TransferRoots();return p,r.OneShape()
# Matched holes in the existing yaw housing and bearing cap. Two fasteners resist bracket twist.
for name,radius in [('yaw-upper-leg-1',1.25),('pitch-bearing-cap-1',1.65)]:
 p,a=load(name)
 for z in [-5,1]:a=cut(a,cy(radius,[10,38.9,z],[0,1,0],11.2))
 for x in [-12,12]:
  if name=='yaw-upper-leg-1':a=fuse(a,cy(4,[x,39,4],[0,1,0],6));a=cut(a,cy(1.25,[x,38.9,4],[0,1,0],6.2))
  else:a=cut(a,cy(3.25,[x,44.9,4],[0,1,0],1.7))
 parts.remove(p);save(name,a,p['parent'],p['xy'],p['material'],p['notes']+' v8 bracket two M3 at localX10 Z-5/+1. Two M3x10 through support lug and cap.')
p,a=load('pitch-yoke-v7');px0,py0=p['xy']
rr=STEPControl_Reader();rr.ReadFile('outputs/danya-simulator/assets/buildable-bust-v7/pitch-yoke-v7.step');rr.TransferRoots();a=rr.OneShape()
for x,y in [(7,36),(19.5,24)]:
 for z in [8,14]:a=cut(a,cy(1.25,[x-px0-5,y-py0,z],[1,0,0],10))
parts.remove(p);save('pitch-yoke-v7',a,p['parent'],p['xy'],p['material'],p['notes']+' v8 M3 bracket holes at worldY36 and24, Z8/14, 6mm pitch. Support metal, not belt tension through plastic shell.')
# Four chassis spacers provide the clearance required by the moving projector tray.
for i,r in enumerate(json.load(open(out/'support-routes.json'))):
 if r.get('status')=='FOUND' and r['name'].startswith(('base-','yaw-motor-')):
  x,y=r['points'][-1][:2];a=cut(cyl(3,-155,4,x,y),cyl(1.65,-155.1,4.2,x,y));save('floor-spacer-'+str(i),a,'fixed',[0,0],'steel','OD6 ID3.3 L4 between floor plate and support foot, M3x16 stack with washer/nyloc.')
(out/'parts.json').write_text(json.dumps({'status':'v8 prototype assembly, strength not tested','parts':parts},indent=2))
