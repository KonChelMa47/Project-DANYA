exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];manifest=json.load(open(out/'drop-structural-candidate.json'));meta={p['name']:p for p in manifest['parts']}
drive=json.load(open(out/'trial-drive-layout.json'));d=drive['motors'][2];old=qm(d['quaternion'],d['tip']);chosen=next(r for r in json.load(open('work/buildable-bust-v10/roll-housing-alt-search.json'))['rows']if not r['hits']);d['quaternion']=chosen['quaternion'];m=qm(d['quaternion'],d['tip'])
def tr(mat):
 t=gp_Trsf();t.SetValues(mat[0],mat[4],mat[8],mat[12],mat[1],mat[5],mat[9],mat[13],mat[2],mat[6],mat[10],mat[14]);return t
change=tr(m).Multiplied(tr(old).Inverted())
def read(n):
 r=STEPControl_Reader();assert r.ReadFile(str(out/(n+'.step')))==1;r.TransferRoots();a=r.OneShape();x,y=meta[n]['xy'];t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));return BRepBuilderAPI_Transform(a,t,True).Shape()
def chain(pts):
 a=None
 for u,v in zip(pts,pts[1:]):
  delta=[v[i]-u[i]for i in range(3)];b=cy(3,u,delta,math.sqrt(sum(x*x for x in delta)));a=b if a is None else fuse(a,b)
 for p in pts[1:-1]:a=fuse(a,BRepPrimAPI_MakeSphere(gp_Pnt(*p),3).Shape())
 return a
moved=[n for n in meta if n.startswith(('selected-backmount-roll-','trial-roll-frame-bolt-'))]
for n in moved:
 p=meta[n];save('revised-'+n,BRepBuilderAPI_Transform(read(n),change,True).Shape(),'pitch',[0,0],p['material'],p['notes']+' Housing rotated20degrees around unchanged output axis. Clearance candidate.')
belt=next(r for r in json.load(open('work/buildable-bust-v10/inward-roll-search.json'))['rows']if all(not c['hits']for c in r['checks']));y3=belt['nodes'][3]['x']+.3457450866699219;z3=belt['nodes'][3]['y'];end=point([20.5,-50,0],m)
for side,x in [('inner',5),('outer',22.5)]:
 pts=[[4.5,36,10],[5,59,-3],[5,60.345745,-10],[5,y3,z3]] if side=='inner' else [[22.5,24,12],[22.5,60.345745,-10],[22.5,y3,z3]]
 a=chain(pts)
 for y,z in [(36 if side=='inner' else 24,10 if side=='inner' else 12),(60.345745,-10),(y3,z3)]:a=fuse(a,cy(6,[x-1.5,y,z],[1,0,0],3))
 a=fuse(a,box(3 if side=='inner'else 21,32 if side=='inner'else 20,4,3,8,13))
 for z in [8,14]:a=cut(a,cy(1.65,[0,36 if side=='inner'else 24,z],[1,0,0],28))
 for y,z in [(60.345745,-10),(y3,z3)]:a=cut(a,cy(2.65,[x-6,y,z],[1,0,0],12))
 if side=='outer':
  a=fuse(a,chain([[22.5,60.345745,-10],[62.5,55.345745,-40],end]));a=fuse(a,move(box(12,-48.6,-10,17,6,20),m));a=cut(a,move(box(-5,-42.6,-18,50,50,36),m))
  for z in [-5,5]:a=cut(a,move(cy(1.7,[20.5,-75,z],[0,1,0],32.5),m));a=cut(a,move(cy(3.2,[20.5,-75,z],[0,1,0],28.4),m))
 a=cut(a,read('pitch-yoke-v7'))
 if side=='outer':
  # Ream after joining the motor branch so the branch cannot refill the axle hole.
  for yy,zz in [(60.34574508666992,-10),(y3,z3)]:a=cut(a,cy(2.65,[0,yy,zz],[1,0,0],30))
  # Retain mounting holes while clearing the body throughout motion.
  tool=poly([(11.9,10.1),(16.1,10.1),(11.9,5.9)],0,8);tt=gp_Trsf();tt.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(1,0,0)),math.pi/2);tool=BRepBuilderAPI_Transform(tool,tt,True).Shape();tt=gp_Trsf();tt.SetTranslation(gp_Vec(0,-41.6,0));tool=BRepBuilderAPI_Transform(tool,tt,True).Shape();a=cut(a,move(tool,m))
 save('revised-roll-'+side,a,'pitch',[0,0],'metal6mm route envelope','Changed idler route and20degree housing. Real rod/printed joint fabrication details and structural qualification pending; do not print entire6mm rod assembly as polymer.')
save('revised-roll-idler-axle-3',cy(2.5,[3,y3,z3],[1,0,0],23),'pitch',[0,0],'M5 smooth shoulder axle envelope','Two-sided support, unchanged23mm span; end retention hardware not yet detailed.')
# Preserve later projector, pitch, floor and lift revisions in the target manifest.
manifest=json.load(open(out/'revised-structural-candidate.json'))
remove=set(moved+['trial-roll-support','roll-idler-carrier-inner','roll-idler-axle-3']);manifest['parts']=[p for p in manifest['parts']if p['name']not in remove and not p['name'].startswith('revised-roll-')and not p['name'].startswith('revised-selected-backmount-roll-')and not p['name'].startswith('revised-trial-roll-frame-bolt-')]+parts
(out/'revised-structural-candidate.json').write_text(json.dumps(manifest,indent=2));(out/'revised-drive-layout.json').write_text(json.dumps(drive,indent=2));(out/'revised-roll-parts.json').write_text(json.dumps({'parts':parts,'status':'candidate, fabrication/clearance not qualified'},indent=2));print('VALID',len(parts))
