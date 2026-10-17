import fs from 'node:fs';
import{BufferGeometry,Float32BufferAttribute,Mesh}from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import{STLExporter}from '../../outputs/danya-simulator/node_modules/three/examples/jsm/exporters/STLExporter.js';
const out='outputs/danya-simulator/assets/buildable-bust-v1';fs.mkdirSync(out,{recursive:true});
// X front/back; Y shoulder width. Independent styling prototype, not a collision-approved replacement.
const profiles=[[-182,88,111,2],[-145,96,126,2],[-100,103,149,2],[-62,99,165,2],[-35,89,162,2],[-12,70,133,2],[12,52,82,2],[38,48,53,2]];
function profile(z){let i=0;while(i<profiles.length-2&&z>profiles[i+1][0])i++;const h=profiles[i+1][0]-profiles[i][0],t=(z-profiles[i][0])/h;return[1,2,3].map(k=>{const sec=j=>(profiles[j+1][k]-profiles[j][k])/(profiles[j+1][0]-profiles[j][0]);const slope=j=>{if(j===0)return sec(0);if(j===profiles.length-1)return sec(j-1);const a=sec(j-1),b=sec(j);return a*b<=0?0:2*a*b/(a+b);};return(2*t**3-3*t*t+1)*profiles[i][k]+(t**3-2*t*t+t)*h*slope(i)+(-2*t**3+3*t*t)*profiles[i+1][k]+(t**3-t*t)*h*slope(i+1);});}

function point(z,t,inner){let[a,b,n]=profile(z);const c=Math.cos(t),s=Math.sin(t),y=b*Math.sign(s)*Math.abs(s)**(2/n),front=Math.max(0,c),back=Math.max(0,-c);
// Paired pectorals, central sternum valley, clavicular ridge and upper-back volume.
const sq=x=>x*x;
const pec=13*Math.exp(-sq((Math.abs(y)-66)/43))*Math.exp(-sq((z+83)/42));
const sternum=4*Math.exp(-sq(y/18))*Math.exp(-sq((z+75)/55));
const clavicle=5*Math.exp(-sq((z+22+Math.abs(y)*.18)/9))*Math.exp(-sq((Math.abs(y)-57)/50));
const scapula=4*Math.exp(-sq((Math.abs(y)-70)/62))*Math.exp(-sq((z+55)/55));
let x=15.7629+a*Math.sign(c)*Math.abs(c)**(2/n)+front**2*(pec-sternum+clavicle)-back**2*scapula;
let yy=y;if(inner){const e=.001,pa=point(z,t+e,false),pb=point(z,t-e,false),pc=point(z+e,t,false),pd=point(z-e,t,false),a=pa.map((v,i)=>v-pb[i]),b=pc.map((v,i)=>v-pd[i]),nn=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],len=Math.hypot(...nn),cosine=(nn[0]*c+nn[1]*s)/len,offset=3/Math.max(.15,cosine);x-=offset*c;yy-=offset*s;}return[x,yy,z];}

const allPositions=[],allNormals=[],manifest={status:'STYLING / SPLIT-PRINT FIT PROTOTYPE ONLY. Not an assembled robot release. Motor mounts, joins, ports, actual wall normal thickness, neck clearance and thermal validation remain.',profiles,assumedBuildVolumeMM:[220,220,250],parts:[]};
for(let band=0;band<2;band++)for(let q=0;q<4;q++){
 const z0=band===0?-182:-95,z1=band===0?-95:38,Z=band===0?30:48,A=32,pos=[],idx=[];
 for(let layer=0;layer<2;layer++)for(let j=0;j<=Z;j++)for(let i=0;i<=A;i++)pos.push(...point(z0+(z1-z0)*j/Z,(q+i/A)*Math.PI/2,layer===1));
 const at=(l,j,i)=>l*(Z+1)*(A+1)+j*(A+1)+i,tri=(a,b,c)=>idx.push(a,b,c),quad=(a,b,c,d)=>{tri(a,b,c);tri(a,c,d);};
 for(let j=0;j<Z;j++)for(let i=0;i<A;i++){quad(at(0,j,i),at(0,j,i+1),at(0,j+1,i+1),at(0,j+1,i));quad(at(1,j,i),at(1,j+1,i),at(1,j+1,i+1),at(1,j,i+1));}
 for(let j=0;j<Z;j++){quad(at(0,j,0),at(0,j+1,0),at(1,j+1,0),at(1,j,0));quad(at(0,j,A),at(1,j,A),at(1,j+1,A),at(0,j+1,A));}
 for(let i=0;i<A;i++){quad(at(0,0,i),at(1,0,i),at(1,0,i+1),at(0,0,i+1));quad(at(0,Z,i),at(0,Z,i+1),at(1,Z,i+1),at(1,Z,i));}
 const g=new BufferGeometry().setAttribute('position',new Float32BufferAttribute(pos,3)).setIndex(idx);g.computeVertexNormals();g.computeBoundingBox();
 const edges=new Map();for(let i=0;i<idx.length;i+=3)for(let k=0;k<3;k++){const a=idx[i+k],b=idx[i+(k+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;edges.set(key,(edges.get(key)||0)+1);}
 const name=`shell-${band?'upper':'lower'}-${q}`,size=g.boundingBox.getSize({set(x,y,z){this.x=x;this.y=y;this.z=z;return this;},subVectors(a,b){return this.set(a.x-b.x,a.y-b.y,a.z-b.z);}}),badEdges=[...edges.values()].filter(n=>n!==2).length;
 const stl=new STLExporter().parse(new Mesh(g),{binary:true});fs.writeFileSync(`${out}/${name}.stl`,Buffer.from(stl.buffer));
 const ni=g.toNonIndexed(),flat=ni.attributes.position.array;allPositions.push(...flat);allNormals.push(...ni.attributes.normal.array);
 manifest.parts.push({name,sizeMM:[size.x,size.y,size.z],boundaryOrNonmanifoldEdges:badEdges,fitsAssumedBuildVolume:size.x<=220&&size.y<=220&&size.z<=250});
}
fs.writeFileSync(`${out}/bust.normals.bin`,Buffer.from(new Float32Array(allNormals).buffer));
fs.writeFileSync(`${out}/bust.positions.bin`,Buffer.from(new Float32Array(allPositions).buffer));fs.writeFileSync(`${out}/bust.indices.bin`,Buffer.from(Uint32Array.from({length:allPositions.length/3},(_,i)=>i).buffer));
manifest.outerSizeMM=[0,1,2].map(k=>{let lo=Infinity,hi=-Infinity;for(let i=k;i<allPositions.length;i+=3){lo=Math.min(lo,allPositions[i]);hi=Math.max(hi,allPositions[i]);}return hi-lo;});manifest.proportionBasis={headWidthMM:159.065,shoulderWidthMM:330,shoulderToHeadRatio:330/159.065,scope:"Artistic muscular-adult-male proportion target, not a personalized anthropometric measurement"};manifest.neckTopSizeMM=[96,106];fs.writeFileSync(`${out}/manifest.json`,JSON.stringify(manifest,null,2));console.log(manifest.parts);
