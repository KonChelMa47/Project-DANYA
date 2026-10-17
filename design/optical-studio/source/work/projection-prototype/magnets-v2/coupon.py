exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/projection-prototype/magnets-v2');parts=[]
a=box(0,0,0,56,16,5.5)
for j,d in enumerate([6.0,6.15,6.3,6.45]):
 x=7+14*j;y=9;a=cut(a,cyl(d/2,2.2,3.4,x,y));a=cut(a,cyl(d/2+.3,3.4,1,x,y))
 for t in [0,120,240]:a=fuse(a,cyl(.55,2.1,.35,x+1.6*math.cos(math.radians(t)),y+1.6*math.sin(math.radians(t))))
 for k in range(j+1):a=cut(a,cyl(.55,5.1,.5,x-(j)*1.2+k*2.4,2))
save('magnet-fit-coupon',a,'fixture',[0,0],'PETG','From1to4dimples: bore6.00,6.15,6.30,6.45mm. Nominalintegratedshellbore6.15. Dryfitfirst. Magnetrecess0.15; supportlands0.25mm; epoxykeyradial0.30mm. Not installed inrobot.')
