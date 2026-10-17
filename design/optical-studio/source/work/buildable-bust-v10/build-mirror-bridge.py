# Printed bridge. Long side struts are not used to mount the main mirror.
exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');parts=[];m=dat['mirror']['matrix']
a=box(-21,-7,-21.1,42,14,5.5)
# Four broad rearward roots, later joined to conformal pads on the head shell.
for x in [-16,16]:
 for y in [-3.5,3.5]:a=fuse(a,cyl(4.5,-31,11,x,y))
for x in [-10,10]:a=cut(a,cyl(1.7,-21.2,5.8,x,0))
sv('mirror-head-bridge',a,'roll',m,'printed PETG candidate','Mounted behind mirror fixed plate, two M3x12 screws. Four integral rear roots to conformal head pads. Whole head attachment union pending; no front-side struts.')
for i,x in enumerate([-10,10]):
 a=fuse(cyl(1.5,-21.6,12,x,0),cyl(2.75,-24.6,3,x,0));sv('mirror-head-bolt-'+str(i),a,'roll',m,'purchased M3x12','Head behind support; thread omitted. Tip clearance to moving mirror backing to be checked.')
 a=cut(cyl(3.5,-21.6,.5,x,0),cyl(1.6,-21.7,.7,x,0));sv('mirror-head-washer-'+str(i),a,'roll',m,'purchased M3 washer','OD7, ID3.2, t0.5.')
(out/'mirror-head-bridge.json').write_text(json.dumps({'status':'candidate','parts':parts},indent=2))
