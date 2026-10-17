exec(open('work/buildable-bust-v8/build-mounts.py').read().split('# Split motor saddles')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
parts=json.load(open(out/'parts.json'))['parts'];parts=[p for p in parts if not p['name'].startswith(('frame-','head-mount-','mirror-adjust-bolt','foldMirror-adjust-bolt'))]
def tube(a,b,r=3):
 v=[b[i]-a[i]for i in range(3)];L=math.sqrt(sum(x*x for x in v));return cy(r,a,v,L)
def chain(pts,r=3):
 a=tube(pts[0],pts[1],r)
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),r).Shape())
 for i in range(2,len(pts)):a=fuse(a,tube(pts[i-1],pts[i],r))
 return a
routes=json.load(open('work/buildable-bust-v8/search-requests-result.json'))
more=json.load(open('work/buildable-bust-v8/search-more-result.json'));routes +=[r for r in more if r['status']=='FOUND' and r['name']!='yaw-motor-support']
finalpath=Path('work/buildable-bust-v8/search-final-result.json')
if finalpath.exists():routes +=[r for r in json.load(open(finalpath))if r['status']=='FOUND']
if Path('work/buildable-bust-v8/search-base3-result.json').exists():routes +=[r for r in json.load(open('work/buildable-bust-v8/search-base3-result.json'))if r['status']=='FOUND']
for f in ['search-refine-result.json','search-pitch-root-result.json','search-main-last-result.json','search-base-last-result.json','search-base-top-result.json']:
 if Path('work/buildable-bust-v8/'+f).exists():
  for new in json.load(open('work/buildable-bust-v8/'+f)):
   if new['status']=='FOUND':routes=[r for r in routes if r['name']!=new['name']]+[new]
# Export formed 6mm metal supports with attachment lugs. These are not plastic rods.
for r in routes:
 if r.get('status')!='FOUND':continue
 a=chain(r['points']);start,end=r['points'][0],r['points'][-1]
 if r['name'].startswith(('main-','projector-')):
  a=fuse(a,cyl(5,43,3,*start[:2]));a=cut(a,cyl(1.65,42.9,3.2,*start[:2]))
 if r['name'].startswith('base-'):
  a=fuse(a,cyl(5,start[2]-(3 if r.get('anchor')=='upper' else 0),3,*start[:2]));a=cut(a,cyl(1.65,start[2]-3.1,6.2,*start[:2]))
 if r['name'].startswith(('base-','yaw-motor-')):
  a=fuse(a,cyl(6,-151,3,*end[:2]));a=cut(a,cyl(1.65,-151.1,3.2,*end[:2]))
 if r['name']=='pitch-motor-support':
  a=fuse(a,box(start[0]-4,start[1]-3,-9,8,3,14))
  for z in [-5,1]:a=cut(a,cy(1.65,[start[0],start[1]-3.1,z],[0,1,0],6.2))
 if r['name']=='roll-motor-support':
  a=fuse(a,box(21,20,4,3,8,13))
  for z in [8,14]:a=cut(a,cy(1.65,[20.9,24,z],[1,0,0],6.2))
 # End lugs at mirror/tray interfaces, aligned to their fastening normals.
 target='mirror' if r['name'].startswith('main-') else 'projector' if r['name'].startswith('projector-') else 'foldMirror' if r['name']=='fold-support' else None
 if target:
  item=next(i for i in json.load(open('work/buildable-bust-v8/interfaces.json'))if i['name']==target);axis=item['axis'];a=fuse(a,cy(4.5,end,axis,3));a=cut(a,cy(1.65,[end[i]-.1*axis[i]for i in range(3)],axis,3.4))
 if r['name']=='roll-motor-support':
  rr=STEPControl_Reader();rr.ReadFile(str(out/'roll-idler-carrier-outer.step'));rr.TransferRoots();a=fuse(a,rr.OneShape());parts=[p for p in parts if p['name']!='roll-idler-carrier-outer']
  a=cut(a,box(-100,20,17,200,8,7));a=cut(a,box(-100,20,4,121,8,13))
 save('frame-'+r['name'],a,r['parent'],[0,0],'steel rod6 with metal lugs','Formed/fabricated support, dimensions in assembled position. Lug holes3.3 for M3 fasteners. Metal fabrication, not an unreinforced FDM rod. Proof test required.')
# Add matching threaded bosses to roll carrier; outside head shape remains untouched.
p=next(p for p in parts if p['name']=='roll-carrier-v7');rr=STEPControl_Reader();rr.ReadFile(str(out/'roll-carrier-v7.step'));rr.TransferRoots();a=rr.OneShape();cx0,cy0=p['xy']
for x,y in [(-48,-29),(-48,30),(4,-29)]:a=fuse(a,cyl(5,37,6,x-cx0,y-cy0));a=cut(a,cyl(1.25,36.9,6.2,x-cx0,y-cy0))
parts.remove(p);save('roll-carrier-v7',a,'roll',[cx0,cy0],'6061-T6 aluminium','v8 three tappedM3 mounting bosses. M3x8 through 3mm frame lug, 5mm nominal engagement.')
# Removable floor chassis, with M3 holes matching support feet.
pts=[(-7.67+80*math.cos(i*math.pi/48),.3457+109*math.sin(i*math.pi/48))for i in range(96)]
a=poly(pts,-159,4);a=cut(a,box(-37,-35,-159.1,60,70,4.2))
for r in routes:
 if r.get('status')=='FOUND' and r['name'].startswith(('base-','yaw-motor-')):
  x,y=r['points'][-1][:2];a=cut(a,cyl(1.65,-159.1,4.2,x,y))
for x,y in [(-75,0),(60,0),(-10,-98),(-10,98)]:a=cut(a,cyl(2.2,-159.1,4.2,x,y))
save('frame-floor',a,'fixed',[0,0],'6061-T6 aluminium4','Floor chassis M3x12 support screws plus washer/nyloc. Four M4 shell mounting holes. Central service opening60x70. Enclosure attachment bosses must match these coordinates.')
# Four compliant head shoes on the internal rear shell. No external head reshaping.
for pad in json.load(open('work/buildable-bust-v8/head-pads.json')):
 p,n=pad['point'],pad['normal'];back=[p[i]-5*n[i]for i in range(3)];a=cy(6,back,n,3.5);a=cut(a,cy(1.25,[back[i]-.1*n[i]for i in range(3)],n,3.7));save('head-mount-'+pad['name'],a,'roll',[0,0],'aluminium pad','Internal bonded shoe, 1.5mm compliant bondline to actual curved shell. M3 tapped mounting. Adhesive strength not yet proven; inspect fit to physical print.')
 # Branch from corresponding main rail near Z65 or100 to shoe back face.
 sign=-1 if p[1]<0 else 1;main=next(r for r in routes if r['name']==('main-left'if sign<0 else'main-right'));u=(p[2]-main['points'][0][2])/(main['points'][1][2]-main['points'][0][2]);q=[main['points'][0][i]+u*(main['points'][1][i]-main['points'][0][i])for i in range(3)];end=[back[i]-3*n[i]for i in range(3)];a=chain([q,end]);a=fuse(a,cy(4.5,end,n,3));a=cut(a,cy(1.65,[end[i]-.1*n[i]for i in range(3)],n,3.2));save('frame-'+pad['name'],a,'roll',[0,0],'steel rod6 with metal lug','Branch to internal head shoe, M3x6 connection; rod intersection requires fabricated joint.')
# Joined metal subassemblies: coincident lugs/branches are one fabricated part, not overlapping loose parts.
for group,names in [('frame-optics-left',['frame-main-left','frame-projector-left','frame-fold-support','frame-head-pad-55--1','frame-head-pad-70--1']),('frame-optics-right',['frame-main-right','frame-head-pad-55-1','frame-head-pad-70-1'])]:
 a=None
 for name in names:
  rr=STEPControl_Reader();rr.ReadFile(str(out/(name+'.step')));rr.TransferRoots();a=rr.OneShape() if a is None else fuse(a,rr.OneShape())
 parts=[p for p in parts if p['name']not in names]
 save(group,a,'roll',[0,0],'fabricated steel rod6 and lugs','Joined subassembly; no overlapping loose flanges. Requires metal fabrication and dimensional jig. Costs not yet quoted; do not substitute thin printed rods.')
(out/'parts.json').write_text(json.dumps({'status':'v8 prototype assembly; checks pending','parts':parts},indent=2));(out/'support-routes.json').write_text(json.dumps(routes,indent=2));print('parts',len(parts))
