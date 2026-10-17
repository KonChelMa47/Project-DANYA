from pathlib import Path
p=Path('outputs/danya-simulator/src/app.js');s=p.read_text();s=s.replace("geometry[key].computeBoundingSphere();","geometry[key].computeBoundingSphere();geometry[key].computeBoundingBox();")
a=s.index('  surfacePoints=[];const scale=state.modelScale;');b=s.index('\n}',a)
s=s[:a]+'''  surfacePoints=[];const scale=state.modelScale,box=geometry.face.boundingBox,step=2*scale;
  // Include the entire face geometry, including any redesigned rim. Never drop occluded targets.
  for(let z=box.min.z+step*.17;z<=box.max.z;z+=step)for(let y=box.min.y+step*.23;y<=box.max.y;y+=step){const ray=new THREE.Ray(new THREE.Vector3(box.max.x+500*scale,y,z),new THREE.Vector3(-1,0,0)),hit=faceBVH.raycastFirst(ray,THREE.DoubleSide,0,1000*scale);if(hit)surfacePoints.push(hit.point);}
'''+s[b:]
s=s.replace("  let surfaceLit=0;for(const point of surfacePoints){const path=projectionPath(snapshot,point);if(path&&visiblePath(snapshot,point,h,f,path))surfaceLit++;}","  let surfaceLit=0;const bounds={min:[1,1],max:[0,0]};for(const point of surfacePoints){const path=projectionPath(snapshot,point);if(path){bounds.min[0]=Math.min(bounds.min[0],path.u);bounds.min[1]=Math.min(bounds.min[1],path.v);bounds.max[0]=Math.max(bounds.max[0],path.u);bounds.max[1]=Math.max(bounds.max[1],path.v);if(visiblePath(snapshot,point,h,f,path))surfaceLit++;}}")
s=s.replace("${surfaceCoverage.toFixed(1)}<small>%</small>","${surfaceLit===surfacePoints.length?'100.0':Math.min(99.9,surfaceCoverage).toFixed(1)}<small>%</small>")
s=s.replace("正面2 mm格子 · 画像光線の到達率 ${pct(counts.face)}%","正面2 mm格子 ${surfaceLit}/${surfacePoints.length}点 · 輪郭外の画角余裕 ${(100*Math.min(...bounds.min,1-bounds.max[0],1-bounds.max[1])).toFixed(1)}%")
s=s.replace("surfaceSamples:surfacePoints.length,mirrorCollision", "surfaceSamples:surfacePoints.length,uvBounds:bounds,mirrorCollision")
p.write_text(s)
