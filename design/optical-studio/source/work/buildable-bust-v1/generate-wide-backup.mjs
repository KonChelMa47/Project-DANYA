import fs from 'node:fs';
import{BufferGeometry,Float32BufferAttribute,Mesh}from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import{STLExporter}from '../../outputs/danya-simulator/node_modules/three/examples/jsm/exporters/STLExporter.js';
const out='outputs/danya-simulator/assets/buildable-bust-v1';fs.mkdirSync(out,{recursive:true});
// X front/back; Y shoulder width. Independent styling prototype, not a collision-approved replacement.
const profiles=[[-240,143,155,2.5],[-180,156,174,2.6],[-105,163,192,2.8],[-48,158,211,2.8],[-8,153,215,2.6],[8,141,178,2.5],[24,90,102,2.2],[38,56,62,2]];
function profile(z){let i=0;while(i<profiles.length-2&&z>profiles[i+1][0])i++;const h=profiles[i+1][0]-profiles[i][0],t=(z-profiles[i][0])/h;return[1,2,3].map(k=>{const sec=j=>(profiles[j+1][k]-profiles[j][k])/(profiles[j+1][0]-profiles[j][0]);const slope=j=>{if(j===0)return sec(0);if(j===profiles.length-1)return sec(j-1);const a=sec(j-1),b=sec(j);return a*b<=0?0:2*a*b/(a+b);};return(2*t**3-3*t*t+1)*profiles[i][k]+(t**3-2*t*t+t)*h*slope(i)+(-2*t**3+3*t*t)*profiles[i+1][k]+(t**3-t*t)*h*slope(i+1);});}

function point(z,t,inner){let[a,b,n]=profile(z);const c=Math.cos(t),s=Math.sin(t),y=b*Math.sign(s)*Math.abs(s)**(2/n),front=Math.max(0,c),back=Math.max(0,-c);
// Paired pectorals, central sternum valley, clavicular ridge and upper-back volume.
const sq=x=>x*x;
const pec=16*Math.exp(-sq((Math.abs(y)-84)/48))*Math.exp(-sq((z+63)/40));
const sternum=4*Math.exp(-sq(y/18))*Math.exp(-sq((z+62)/56));
const clavicle=5*Math.exp(-sq((z+10+Math.abs(y)*.10)/10))*Math.exp(-sq((Math.abs(y)-76)/62));
const scapula=7*Math.exp(-sq((Math.abs(y)-70)/62))*Math.exp(-sq((z+55)/55));
let x=15.7629+a*Math.sign(c)*Math.abs(c)**(2/n)+front**2*(pec-sternum+clavicle)-back**2*scapula;
let yy=y;if(inner){x-=3*Math.sign(c)*Math.abs(c)**(2/n);yy-=3*Math.sign(s)*Math.abs(s)**(2/n);}return[x,yy,z];}

const allPositions=[],allNormals=[],manifest={status:'STYLING / SPLIT-PRINT FIT PROTOTYPE ONLY. Not an assembled robot release. Motor mounts, joins, ports, actual wall normal thickness, neck clearance and thermal validation remain.',profiles,assumedBuildVolumeMM:[220,220,250],parts:[]};
for(let band=0;band<2;band++)for(let q=0;q<4;q++){
 const z0=band===0?-240:-95,z1=band===0?-95:38,Z=band===0?30:48,A=32,pos=[],idx=[];
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
manifest.outerSizeMM=[0,1,2].map(k=>{let lo=Infinity,hi=-Infinity;for(let i=k;i<allPositions.length;i+=3){lo=Math.min(lo,allPositions[i]);hi=Math.max(hi,allPositions[i]);}return hi-lo;});manifest.proportionBasis={headWidthMM:159.065,shoulderWidthMM:430,shoulderToHeadRatio:430/159.065,scope:"Artistic muscular-adult-male proportion target, not a personalized anthropometric measurement"};manifest.neckTopSizeMM=[112,124];fs.writeFileSync(`${out}/manifest.json`,JSON.stringify(manifest,null,2));console.log(manifest.parts);
