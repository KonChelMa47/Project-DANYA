import {Shape,ShapeGeometry,ExtrudeGeometry,BufferGeometry,Float32BufferAttribute} from 'three';
export function mirrorShape(mirror){return new Shape((mirror.outline||[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).map(([x,y])=>({x,y})));}
export function mirrorSurfaceGeometry(mirror){if(mirror.curvatureRadius)return curvedGeometry(mirror,false);return new ShapeGeometry(mirrorShape(mirror));}
export function mirrorSolidGeometry(mirror){if(mirror.curvatureRadius)return curvedGeometry(mirror,true);const g=new ExtrudeGeometry(mirrorShape(mirror),{depth:1.5,bevelEnabled:false,steps:1});g.translate(0,0,-.75);return g;}
function curvedGeometry(mirror,solid){
 const R=mirror.curvatureRadius,w=mirror.width,h=mirror.height,poly=mirror.outline||[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],rings=32,n=poly.length,positions=[],indices=[];
 const make=(fraction,offset)=>{positions.push(0,0,solid?.75+offset:0);for(let ring=1;ring<=rings;ring++)for(const[a,b]of poly){const x=a*w*ring/rings,y=b*h*ring/rings,z=Math.sqrt(Math.max(0,R*R-x*x-y*y));positions.push(x*fraction/w,y*fraction/h,z*fraction-R+(solid?.75:0));}};
 make(1,0);const layer=positions.length/3;
 const cap=(base,reverse)=>{const tri=(a,b,c)=>indices.push(...(reverse?[a,c,b]:[a,b,c]));for(let j=0;j<n;j++)tri(base,base+1+j,base+1+(j+1)%n);for(let k=1;k<rings;k++)for(let j=0;j<n;j++){const a=base+1+(k-1)*n+j,b=base+1+(k-1)*n+(j+1)%n,c=base+1+k*n+j,d=base+1+k*n+(j+1)%n;tri(a,c,d);tri(a,d,b);}};
 cap(0,false);if(solid){make(1-1.5/R,-1.5);cap(layer,true);for(let j=0;j<n;j++){const a=1+(rings-1)*n+j,b=1+(rings-1)*n+(j+1)%n;indices.push(a,a+layer,b+layer,a,b+layer,b);}}
 const g=new BufferGeometry();g.setAttribute('position',new Float32BufferAttribute(positions,3));g.setIndex(indices);g.computeVertexNormals();return g;
}
