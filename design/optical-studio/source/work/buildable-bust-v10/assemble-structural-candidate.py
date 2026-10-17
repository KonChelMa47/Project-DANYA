exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
import shutil
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');old=out.with_name('buildable-bust-v8');v9=out.with_name('buildable-bust-v9');parts=[]
def copy(p,src):
 parts.append(p)
 for ext in ['.stl','.step']:
  if (src/(p['name']+ext)).exists():shutil.copy2(src/(p['name']+ext),out/(p['name']+ext))
for p in json.load(open(v9/'parts.json'))['parts']:
 if not p['name'].startswith('selected-backmount-'):copy(p,v9)
for file in ['motor-backplates-v10.json','yaw-rear-stand.json']:
 parts+=json.load(open(out/file))['parts']
parts.append({'name':'mirror-head-cradle','parent':'roll','xy':[0,0],'material':'printed PETG candidate','notes':'Conformal bonded cradle replaces long mirror struts. Bond strength not tested.'})
parts+=json.load(open(out/'mirror-head-bridge.json'))['parts'][1:]
for p in json.load(open(old/'parts.json'))['parts']:
 n=p['name']
 if n.startswith('head-mount-') or n in ['frame-projector-right','frame-base-right','frame-base-left','frame-base-third','frame-pitch-motor-support','frame-roll-motor-support']:copy(p,old)
 if n in ['frame-optics-left','frame-optics-right','frame-floor']:
  r=STEPControl_Reader();r.ReadFile(str(old/(n+'.step')));r.TransferRoots();a=r.OneShape()
  if n!='frame-floor':a=cut(a,box(-300,-300,85,600,600,300))
  else:
   for x,y in [(-60,-60),(-40,-60)]:
    a=cut(a,cyl(2.2,-159.1,4.2,x,y));a=cut(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,-159.1),gp_Dir(0,0,1)),4.2,2.2,2.).Shape())
  save(n,a,p['parent'],p['xy'],p['material'],'v10 clearance candidate. Upper mirror rails cut atZ85; mirror supported by head cradle. Legacy frame material/manufacturing remains provisional. Pitch/roll/projector frame interfaces require updated attachment adapters.')
for i,(x,y)in enumerate([(-60,-60),(-40,-60)]):
 a=cut(cyl(4,-155,2,x,y),cyl(2.2,-155.1,2.2,x,y));save('yaw-floor-spacer-'+str(i),a,'fixed',[0,0],'M4 spacer OD8 length2 candidate','Stock hardware envelope; supplier pending.')
 a=fuse(cyl(2,-157,18,x,y),BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x,y,-159),gp_Dir(0,0,1)),4,2,2.).Shape());save('yaw-floor-bolt-'+str(i),a,'fixed',[0,0],'M4x20 countersunk candidate','90degree head envelope, thread omitted.')
 a=cut(cyl(4.5,-147,.8,x,y),cyl(2.2,-147.1,1,x,y));save('yaw-floor-washer-'+str(i),a,'fixed',[0,0],'M4 washer OD9 t0.8','Stock envelope.')
 a=cut(poly([(x+7/math.sqrt(3)*math.cos(k*math.pi/3),y+7/math.sqrt(3)*math.sin(k*math.pi/3))for k in range(6)],-146.2,3.2),cyl(1.7,-146.3,3.4,x,y));save('yaw-floor-nut-'+str(i),a,'fixed',[0,0],'M4 hex nut AF7 t3.2','Thread omitted.')
for name in ['drive-layout.json','transmission.json','pitch-belt.stl','roll-belt.stl']:shutil.copy2(v9/name,out/name)
shutil.copy2(out.with_name('buildable-bust-v5')/'gear-layout.json',out/'gear-layout.json')
(out/'structural-candidate.json').write_text(json.dumps({'status':'FULL COLLISION STUDY candidate; not assembly release. Retained pitch/roll/projector support ends still need new interface adapters; legacy fabrication not all converted. Magnet system excluded until valid.','parts':parts},indent=2));print('STRUCTURAL CANDIDATE',len(parts))
