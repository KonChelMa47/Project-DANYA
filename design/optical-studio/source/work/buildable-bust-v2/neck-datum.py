import numpy as np,json
from pathlib import Path
p=np.fromfile('work/adult-kit/original-head.positions.bin','<f4').reshape(-1,3)
i=np.fromfile('work/adult-kit/head-outer-original.indices.bin','<u4').reshape(-1,3)
z=float(p[:,2].min()+.5);t=p[i];hits=[]
for a,b in [(0,1),(1,2),(2,0)]:
 v,w=t[:,a],t[:,b];m=(v[:,2]<z)!=(w[:,2]<z);v,w=v[m],w[m];hits.append(v+(w-v)*((z-v[:,2])/(w[:,2]-v[:,2]))[:,None])
h=np.concatenate(hits);lo=h.min(0);hi=h.max(0);c=(lo+hi)/2
# Preserve actual asymmetric cross-section rather than fit a generic ellipse.
a=np.arctan2(h[:,1]-c[1],h[:,0]-c[0]);r=np.linalg.norm(h[:,:2]-c[:2],axis=1);order=np.argsort(a);a,r=a[order],r[order];angles=np.linspace(-np.pi,np.pi,257)[:-1];rad=np.interp(angles,np.r_[a[-1]-2*np.pi,a,a[0]+2*np.pi],np.r_[r[-1],r,r[0]])
outline=np.column_stack([c[0]+rad*np.cos(angles),c[1]+rad*np.sin(angles),np.full(256,z)])
o={'method':'Bounding-box center of original head outer section 0.5 mm above bottom; circumferentially interpolated 256-point section. Design datum, not a measured anatomical joint.', 'sectionZ':z,'center':c.tolist(),'bounds':[lo.tolist(),hi.tolist()],'outline':outline.tolist(),'maxAngularGapDegrees':float(np.diff(np.r_[a,a[0]+2*np.pi]).max()*180/np.pi)}
Path('outputs/danya-simulator/assets/buildable-bust-v2/neck-datum.json').write_text(json.dumps(o,indent=2));print({k:v for k,v in o.items() if k!='outline'})
