exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];a=box(0,0,0,60,18,7);holes=[]
for i,d in enumerate([6.0,6.1,6.15,6.2]):
 x=9+14*i;a=cut(a,cyl(d/2,3.7,3.4,x,9));holes.append({'index':i+1,'center':[x,9],'diameterMM':d,'depthMM':3.3})
 for j in range(i+1):a=cut(a,box(x-3+j*1.5,1,6.3,.7,2,.8))
save('magnet-fit-coupon',a,'fixture',[0,0],'printed material same as final seats','One to four engraved marks identifyD6.0/6.1/6.15/6.2 pockets, all3.3deep. Trial fit actual5.9x2.9 magnets without force; printing calibration may require changing selected pocket diameter.')
(out/'magnet-fit-coupon.json').write_text(json.dumps({'status':'fit test coupon, no retention proof','magnetMM':[5.9,2.9],'holes':holes,'parts':parts},indent=2));print('VALID COUPON')
