exec(open('work/printable-mounts-v5/build-tray.py').read().split('# Bottom-up')[0])
def read(p):
 r=STEPControl_Reader();assert r.ReadFile(str(p))==1;r.TransferRoots();return r.OneShape()
tray=read(out/'drop-projector-tray-t5.step');cap=read(out/'tray-nut-cover-local.step');seat=read(out/'tray-nut-fit-coupon.step');rows=[]
# Remove only the spring noses for rigid insertion-path verification.
core=cap
for sg in [-1,1]:
 pts=[(sg*x,y)for x,y in [(4.7,-2.8),(5.15,-2.8),(4.6,-2.25),(4.15,-2.25)]];core=cut(core,xyzpoly(pts,3,1.4))
for i,(x,z,sg)in enumerate([(12,7,1),(12,68,-1),(123,7,1),(123,68,-1)]):
 stand=read(out.parent/'assembly'/f'projector-stand-{i+1}.step');c=read(out/f'tray-nut-cover-{i}.step')
 path=[volume(BRepAlgoAPI_Common(place(core,x,z+sg*dz,sg),tray).Shape())for dz in range(16)]
 rows.append({'seat':i,'trayStandOverlapMM3':volume(BRepAlgoAPI_Common(tray,stand).Shape()),'coverStandOverlapMM3':volume(BRepAlgoAPI_Common(c,stand).Shape()),'nonDetentSlideOverlapMM3':max(path)})
# Hex floor is a solid vertical extrusion starting at bed Y=-8; no roof over hex.
floor=cut(hex_y(0,-8,0,2.35,5.8),cy(1.7,[0,-8.1,0],[0,1,0],2.55))
openColumn=hex_y(0,-5.649,0,10,5.799)
report={'mates':rows,'nutFloorMissingMM3':volume(cut(floor,seat)),'nutLoadingColumnBlockedMM3':volume(BRepAlgoAPI_Common(openColumn,seat).Shape()),'supportFreeNutRegion':'Bed-connected floor 2.35mm; vertical hex walls; open top; dovetail undersides45degrees. Cover printed separately. Physical print not yet tested.'}
# Nut must not pass floor, and cover must not pull upward or backward when latched.
nut=cut(hex_y(0,-5.55,0,2.4,5.5),cy(1.3,[0,-5.65,0],[0,1,0],2.6))
def shift(a,x,y,z):return move(a,[1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1])
report['nutDown0_5BlockedMM3']=volume(BRepAlgoAPI_Common(shift(nut,0,-.5,0),seat).Shape())
report['coverUp0_5BlockedMM3']=volume(BRepAlgoAPI_Common(shift(cap,0,.5,0),seat).Shape())
report['coverWithdrawal1BlockedMM3']=volume(BRepAlgoAPI_Common(shift(cap,0,0,1),seat).Shape())
(out/'tray-mating-check.json').write_text(json.dumps(report,indent=2));print(report)
assert all(max(row[k]for k in ['trayStandOverlapMM3','coverStandOverlapMM3','nonDetentSlideOverlapMM3'])<1e-5 for row in rows)
assert report['nutFloorMissingMM3']<1e-5 and report['nutLoadingColumnBlockedMM3']<1e-5
assert min(report[k]for k in ['nutDown0_5BlockedMM3','coverUp0_5BlockedMM3','coverWithdrawal1BlockedMM3'])>0

