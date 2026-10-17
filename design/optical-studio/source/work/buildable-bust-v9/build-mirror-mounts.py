exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
parts=[];interfaces=[]
for key in ['mirror','foldMirror']:
 o=layout[key];m=dat[key]['matrix'];w,h=o['width'],o['height'];outline=o.get('outline',[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]);pts=[[-.25*w,-.18*h],[.25*w,-.18*h],[0,.25*h]]
 a=poly([(x*(w-2)*(.82 if key=='mirror' else .88),y*(h-2)*(.82 if key=='mirror' else .88))for x,y in outline],-7.6,5.5)
 for j,(x,y)in enumerate(pts):
  t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));pocket=BRepBuilderAPI_Transform(hexagon(5.7,-7.7,2.7),t,True).Shape();a=cut(a,cyl(1.7,-7.7,5.7,x,y));a=cut(a,pocket)
  nut=cut(hexagon(5.5,-7.5,2.4),cyl(1.3,-7.6,2.6));nut=BRepBuilderAPI_Transform(nut,t,True).Shape();sv('printed-'+key+'-adjust-nut-'+str(j),nut,'roll',m,'purchased M3 hex nut','Install from backing rear before mirror assembly.')
  sv('printed-'+key+'-bond-pad-'+str(j),cyl(3 if key=='foldMirror' else 6,-2.1,1,x,y),'roll',m,'silicone adhesive1mm','Mirror back only. Bond test still required; no front edge masking.')
 sv('printed-'+key+'-backing',a,'roll',m,'printed PETG candidate','5.5mm backing with three captive M3 nuts, no tapped plastic. Reflective plane and glass position unchanged.')
 a=poly(pts,-15.6,5)
 for x,y in pts:a=fuse(a,cyl(4 if key=='foldMirror' else 7,-15.6,5,x,y))
 if key=='foldMirror':a=fuse(a,box(-9,-5,-15.6,18,10,5))
 for j,(x,y)in enumerate(pts):
  a=cut(a,cyl(1.7,-15.7,5.2,x,y))
  bolt=fuse(cyl(1.5,-16.1,12,x,y),cyl(2.75,-19.1,3,x,y));sv('printed-'+key+'-adjust-bolt-'+str(j),bolt,'roll',m,'purchased M3x12 socket screw','12mm under-head length. Washer0.5 +fixedplate5 +elasticgap3 +backing nut. Nominal screw tipZ-4.1, glass rearZ-1.1.')
  sv('printed-'+key+'-adjust-washer-'+str(j),cut(cyl(3.5,-16.1,.5,x,y),cyl(1.6,-16.2,.7,x,y)),'roll',m,'purchased M3 washer7x0.5','Hardware envelope.')
  sv('printed-'+key+'-elastic-'+str(j),cut(cyl(3,-10.6,3,x,y),cyl(1.7,-10.7,3.2,x,y)),'roll',m,'silicone sleeve3mm','Calibration preload must be determined physically. Maintain gap>=3mm; no optical power change.')
 fpts=[[-5,0],[5,0]] if key=='foldMirror' else [[-10,0],[10,0]]
 for j,(x,y)in enumerate(fpts):
  a=cut(a,cyl(1.7,-15.7,5.2,x,y));t=gp_Trsf();t.SetTranslation(gp_Vec(x,y,0));a=cut(a,BRepBuilderAPI_Transform(hexagon(5.7,-13.2,2.7),t,True).Shape())
  nut=cut(hexagon(5.5,-13.1,2.4),cyl(1.3,-13.2,2.6));sv('printed-'+key+'-frame-nut-'+str(j),BRepBuilderAPI_Transform(nut,t,True).Shape(),'roll',m,'purchased M3 hex nut','Frame connection via M3x8 through3mm lug; load nut from front before backing assembly.')
 sv('printed-'+key+'-fixed-plate',a,'roll',m,'printed PETG candidate','Three adjustment bolts, two independent M3 frame attachment points. 3mm nominal elastomer gap. Requires printed structural frame.')
 interfaces.append(dict(name=key,parent='roll',matrix=m,point=point([0,0,-15.6],m),holes=[point([x,y,-15.6],m)for x,y in fpts],axis=[m[8],m[9],m[10]]))
(out/'mirror-mounts.json').write_text(json.dumps({'status':'printed mirror mounts; frame and qualification pending','parts':parts},indent=2));(out/'mirror-interfaces.json').write_text(json.dumps(interfaces,indent=2));print('VALID',len(parts))
