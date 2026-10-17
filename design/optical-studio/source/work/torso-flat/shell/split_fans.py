import numpy as np

def split_fans(v,f):
 incident=[[] for _ in range(len(v))]
 for i,tri in enumerate(f):
  for x in tri:incident[x].append(i)
 extra=[];source=list(range(len(v)));f=f.copy()
 for x,faces in enumerate(incident):
  parent=list(range(len(faces)));edge={}
  def find(a):
   while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
   return a
  for j,fi in enumerate(faces):
   for other in f[fi]:
    if other==x:continue
    if int(other) in edge:parent[find(j)]=find(edge[int(other)])
    else:edge[int(other)]=j
  groups={}
  for j,fi in enumerate(faces):groups.setdefault(find(j),[]).append(fi)
  for group in list(groups.values())[1:]:
   new=len(v)+len(extra);extra.append(v[x]);source.append(x)
   for fi in group:f[fi][f[fi]==x]=new
 return np.concatenate([v,np.array(extra).reshape(-1,3)]),f,np.array(source)
