import numpy as np,json,math
from pathlib import Path
v=np.concatenate([np.fromfile('work/torso-flat/shell/bevel-0.6.positions.bin','<f4').reshape(-1,3),np.fromfile('work/torso-flat/face-local-v2.positions.bin','<f4').reshape(-1,3)]).astype(float);pivot=np.array([35,0,-65.]);w=v-pivot;best=1e9;witness=None
for pitch in np.arange(-20,20.01,2.5):
 for roll in np.arange(-15,15.01,2.5):
  p,r=np.deg2rad([pitch,roll]);z=-math.sin(p)*w[:,0]+math.cos(p)*math.sin(r)*w[:,1]+math.cos(p)*math.cos(r)*w[:,2]+pivot[2];i=z.argmin()
  if z[i]<best:best=float(z[i]);witness=dict(pitch=float(pitch),roll=float(roll),point=v[i].tolist())
radius=float(np.linalg.norm(w,axis=1).max());guard=radius/2*(2*math.radians(1.25))**2
out=dict(pivot=pivot.tolist(),sampledMinZMM=best,continuousGuardMM=guard,continuousLowerBoundZMM=best-guard,witness=witness,scope='All head+face vertices. Z unaffected by yaw. Rotation order Rz Ry Rx. Taylor bound using zero derivative at interior extrema and exact sampled endpoints; triangles cannot extend below their vertices. Rigid chest below bound cannot contact head/face. Does not validate flexible cover or internal structure.')
Path('work/buildable-bust-v2/head-floor.json').write_text(json.dumps(out,indent=2));print(out)
