import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root=Path('outputs/danya-simulator');s=json.load(open('work/flat-redesign/candidate-layout.json'))
def rot(v):
 x,y,z=np.radians(v);return np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])@np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])@np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
def hull(v):
 p=sorted(set(map(tuple,v)));cross=lambda a,b,c:(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);lo=[];hi=[]
 for q in p:
  while len(lo)>1 and cross(lo[-2],lo[-1],q)<=0:lo.pop()
  lo.append(q)
 for q in reversed(p):
  while len(hi)>1 and cross(hi[-2],hi[-1],q)<=0:hi.pop()
  hi.append(q)
 return np.array(lo[:-1]+hi[:-1])
fig,ax=plt.subplots(figsize=(8,9),facecolor='#101925');ax.set_facecolor('#101925')
for n,c in [('head','#65788e'),('face','#badfd9')]:
 v=np.fromfile(f'work/adult-kit/original-{n}.positions.bin','<f4').reshape(-1,3);v=v[np.abs(v[:,1])<2];ax.scatter(v[:,0],v[:,2],s=.5,c=c,label=n+' section (110%)')
v=np.fromfile('work/cad-internal/cad-mesh/kit.local.positions.bin','<f4').reshape(-1,3);v=v@rot(s['projector']['rotation']).T+s['projector']['position'];p=hull(v[::3][:,[0,2]]);ax.fill(p[:,0],p[:,1],color='#d99161',alpha=.8,label='Complete projector (side silhouette)')
for k in ['foldMirror','mirror']:
 m=s[k];v=np.array([[x*m['width'],y*m['height'],0]for x,y in m['outline']])@rot(m['rotation']).T+m['position'];p=hull(v[:,[0,2]]);ax.fill(p[:,0],p[:,1],color='#6ab4ff',alpha=.8,label=k)
P=np.array(s['projector']['position']);A=np.array(s['foldMirror']['position']);B=np.array(s['mirror']['position']);Q=np.array([80,0,145]);line=np.array([P,A,B,Q]);ax.plot(line[:,0],line[:,2],c='#cfe8ab',lw=2,ls='--',label='Illustrative route between centers')
zhead=42.1055316;zmin=v.min(0)[2] if False else -139.8827117064726
ax.axhline(zhead,c='#a3afbc',ls=':',lw=1);ax.annotate('',xy=(112,zhead),xytext=(112,zmin),arrowprops=dict(arrowstyle='<->',color='#f1b576'));ax.text(116,(zhead+zmin)/2,'182 mm\nbelow head',color='#f1b576',va='center',fontsize=11)
ax.set_title('PLANAR-MIRROR STUDY — NOT AN APPROVED DESIGN\nCoverage in outer-surface model; fails compactness / full-shell checks',color='white',fontsize=11,pad=16)
ax.set_xlabel('X / front-back (mm)',color='white');ax.set_ylabel('Z / height (mm)',color='white');ax.set_aspect('equal');ax.set_xlim(-125,182);ax.tick_params(colors='#aebed0');ax.grid(alpha=.12);[sp.set_color('#485564') for sp in ax.spines.values()];ax.legend(loc='lower left',fontsize=8,facecolor='#172536',labelcolor='white');fig.tight_layout();fig.savefig(root/'flat-mirror-study.png',dpi=160)
