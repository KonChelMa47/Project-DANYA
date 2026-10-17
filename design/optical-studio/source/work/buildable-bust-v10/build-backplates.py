exec(open('work/buildable-bust-v9/build-motor-candidates.py').read().split('for d,parent')[0])
out=Path('outputs/danya-simulator/assets/buildable-bust-v10');drive=json.load(open(out.with_name('buildable-bust-v9')/'drive-layout.json'));parts=[]
for d,parent in zip(drive['motors'],['fixed','yaw','pitch']):
 m=qm(d['quaternion'],d['tip']);role=d['role'];a=box(5,-42.6,-14,31,4,28);a=cut(a,cy(10.7,[0,-42.7,0],[0,1,0],4.2))
 for x in [8.3,32.75]:
  for z in [-10.25,10.25]:
   a=cut(a,cy(1.1,[x,-42.7,z],[0,1,0],4.2))
   # Hard locating pad at the documented rear mounting-hole face. Case is not squeezed.
   pad=cut(cy(2,[x,-38.6,z],[0,1,0],3.4),cy(1.1,[x,-38.7,z],[0,1,0],3.6));a=fuse(a,pad)
   bolt=fuse(cy(1,[x,-42.9,z],[0,1,0],10),cy(1.75,[x,-44.3,z],[0,1,0],1.4));sv('selected-backmount-'+role+'-screw-'+str(x)+'-'+str(z),bolt,parent,m,'M2x10 candidate','Manufacturer CAD has1.6mm pilot at these rear holes,2.8mm depth. M2 thread/pitch not specified in downloaded drawing; procurement verification required. 0.3 washer and7.4mount stack ->2.3mm penetration, check actual supplied screws before use.')
   washer=cut(cy(2.5,[x,-42.9,z],[0,1,0],.3),cy(1.1,[x,-43,z],[0,1,0],.5));sv('selected-backmount-'+role+'-washer-'+str(x)+'-'+str(z),washer,parent,m,'purchased M2 washer5x0.3','Envelope.')
 for j,z in enumerate([-5,5]):
  a=cut(a,cy(1.7,[20.5,-42.7,z],[0,1,0],4.2));a=cut(a,hex_y(20.5,-41.3,z,2.8))
  nut=cut(hex_y(20.5,-41.2,z,2.4,5.5),cy(1.3,[20.5,-41.3,z],[0,1,0],2.6));sv('selected-backmount-'+role+'-frame-nut-'+str(j),nut,parent,m,'purchased M3 hex nut','Insert nuts from servo-facing side before mounting plate. Rear frame connects using two M3x8 through4mm lugs; tipY-38.6. Tool access and torque qualification pending.')
 sv('selected-backmount-'+role+'-plate',a,parent,m,'printed PETG candidate','4mm plate with3.4mm isolated mounting pads; mainfrontY-38.6 clearscaseback, four existing rear mounting holes. Does not use servo horn as case support. No case clamping. No case disassembly. M2 screw thread confirmation required.')
(out/'motor-backplates-v10.json').write_text(json.dumps({'status':'rear mounting candidate; pilot dimensions from official STEP, actual thread confirmation pending','parts':parts},indent=2));print('VALID',len(parts))
