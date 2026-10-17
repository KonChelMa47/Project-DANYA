exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
from OCP.STEPControl import STEPControl_Reader
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
source=json.load(open(out/'trial-structural-candidate.json'));meta={p['name']:p for p in source['parts']}
def read(n):
 r=STEPControl_Reader();assert r.ReadFile(str(out/(n+'.step')))==1;r.TransferRoots();return r.OneShape()
def shift(a,z):
 t=gp_Trsf();t.SetTranslation(gp_Vec(0,0,z));return BRepBuilderAPI_Transform(a,t,True).Shape()
def world(n):
 a=read(n);x,y=meta[n]['xy'];t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));return BRepBuilderAPI_Transform(a,t,True).Shape()
names=[n for n in meta if n.startswith(('projector-tray-t5','printed-foldMirror','trial-fold-frame-','trial-tray-frame-'))]
for n in names:
 p=meta[n];save('drop-'+n,shift(read(n),-6),p['parent'],p['xy'],p['material'],p['notes']+' Candidate: lowered6mm with projector/first mirror; requires complete revalidation.')
tray=shift(world('projector-tray-t5'),-6)
for n in ['trial-frame-projector-right','trial-frame-optics-left']:
 a=world(n);top=cut(a,box(-300,-300,-300,600,600,327));lower=cut(a,box(-300,-300,34,600,600,300));a=fuse(shift(lower,-6),top);a=cut(a,tray)
 if n.endswith('optics-left'):
  for k in names:
   if k.startswith('printed-foldMirror') and not any(q in k for q in ['bolt','washer','nut','spring']):a=cut(a,shift(world(k),-6))
 save('drop-'+n,a,'roll',[0,0],meta[n]['material'],'Lower optical support extended6mm at upper joint, receiver/fold plate actual volumes removed. Single-solid topology required. Structural qualification and collision clearance pending.')
# Lower floor4.5mm: gives nominal1.45mm below continuous lowest receiver bolt.
p=meta['trial-floor'];save('drop-floor',shift(read('trial-floor'),-4.5),'fixed',p['xy'],p['material'],'Candidate floorZ-163.5..-156.7. Adds4.5mm downward travel; rear feet and shell extension NOT yet connected, so not an assembly release.')
replacements={n:'drop-'+n for n in names+['trial-frame-projector-right','trial-frame-optics-left']};replacements['trial-floor']='drop-floor'
result=[p for p in source['parts']if p['name']not in replacements]+parts
(out/'drop-structural-candidate.json').write_text(json.dumps({'status':'INCOMPLETE candidate:6mm lower projector/fold; rear floor feet/shell extensions pending','opticalDropMM':6,'parts':result},indent=2))
print('VALID',len(parts))
