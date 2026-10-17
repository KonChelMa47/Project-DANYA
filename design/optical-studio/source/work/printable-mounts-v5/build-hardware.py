exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/projection-prototype/printable-mounts-v5');parts=[]
m=json.load(open(out/'head-check.json'))['mirrorMatrix']
for i,(x,y)in enumerate([(-16,-7),(-16,7),(16,-7),(16,7)]):
 sv(f'mirror-cartridge-bolt-{i}',fuse(cyl(1.5,-44.5,16,x,y),cyl(2.75,-47.5,3,x,y)),'static',m,'purchased M3x16 socket screw','Exterior rear access,washer0.5. Tip-28.5; nut front-30.1; open tip clearance. Check actual screw/washer stack.')
 sv(f'mirror-cartridge-washer-{i}',cut(cyl(3.5,-44.5,.5,x,y),cyl(1.7,-44.6,.7,x,y)),'static',m,'M3 washer OD7 t0.5','Recessed exterior washer')
 sv(f'mirror-cartridge-nut-{i}',move(cut(hexagon(5.5,-32.5,2.4),cyl(1.3,-32.6,2.6)),[1,0,0,0,0,1,0,0,0,0,1,0,x,y,0,1]),'static',m,'M3 hex nut AF5.5 t2.4','Front-loaded deep hex well; shoulderZ-32.6 transfers bolt preload into cradle. Insert nuts before mirror assembly.')
(out/'head-hardware.json').write_text(json.dumps({'parts':parts},indent=2))
