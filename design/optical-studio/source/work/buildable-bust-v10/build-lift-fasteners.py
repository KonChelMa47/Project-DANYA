exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[]
for i,(x,y) in enumerate([(-12,-41),(12,-41),(-12,41),(12,41)]):
 xy=[-22.01616859436035+x,.3457450866699219+y]
 save('lift-carrier-spacer-'+str(i),ring(3,1.6,-14.5,14),'yaw',xy,'stainless steel HIROSUGI CU-314 SUS303','CatalogOD6 ID3.2(+/-0.15) L14(+/-0.1). Replaces scaled display spacer. WorldZ-14.5..-0.5, common8mm lift exempt. Fits/load/tolerance proof pending.');parts[-1]['liftExempt']=True
 a=fuse(cyl(1.5,-22.8,28.3),cyl(3,-24.5,.2));a=fuse(a,BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(0,0,-24.3),gp_Dir(0,0,1)),3,1.5,1.5).Shape())
 save('lift-carrier-countersunk-bolt-'+str(i),a,'yaw',xy,'steel M3x30 countersunk screw envelope','Overalllength30 inclhead. BottomZ-24.5, tipZ5.5; nominal6mm engagement in raised leg foot. Thread omitted; exact supplier head geometry and tolerance stack pending.');parts[-1]['liftExempt']=True
(out/'lift-fasteners.json').write_text(json.dumps({'parts':parts,'source':'https://id.c.misumi-ec.com/book/HSK1_02/pdf/0212.pdf','status':'stock spacer selected, screw/tolerance/strength qualification pending'},indent=2));p=out/'revised-structural-candidate.json';d=json.load(open(p));d['parts']=[p for p in d['parts']if not p['name'].startswith(('carrier-standoff-','lift-carrier-'))]+parts;p.write_text(json.dumps(d,indent=2));print('VALID',len(parts))
