import fs from 'node:fs';
import {Vector3,Ray,DoubleSide,BufferGeometry,BufferAttribute,Matrix4,Box3} from '../../outputs/danya-simulator/node_modules/three/build/three.module.js';
import {MeshBVH} from '../../outputs/danya-simulator/node_modules/three-mesh-bvh/build/index.module.js';
import {cadEngine,kit} from '../adult-kit/load-model.mjs';
import {basis,projectionPath,visiblePath,rayForUV} from '../../outputs/danya-simulator/src/optics.js';
import {createStrayTracer} from '../../outputs/danya-simulator/src/stray-light.js';
const s=JSON.parse(fs.readFileSync('work/torso-flat/layout.json'));
const read=(p,T)=>{const b=fs.readFileSync(p);return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const geo=(p,i)=>new BufferGeometry().setAttribute('position',new BufferAttribute(read(p,Float32Array),3)).setIndex(new BufferAttribute(read(i,Uint32Array),1));
const fg=geo('work/adult-kit/original-face.positions.bin','work/adult-kit/original-face-outer.indices.bin'),f=new MeshBVH(fg),h=new MeshBVH(geo('work/adult-kit/original-head.positions.bin','work/adult-kit/head-outer-original.indices.bin'));
const fullHead=new MeshBVH(geo('work/torso-flat/shell/bevel-0.6.positions.bin','work/torso-flat/shell/bevel-0.6.indices.bin')),fullFace=new MeshBVH(geo('work/torso-flat/face-local-v2.positions.bin','work/torso-flat/face-local-v2.indices.bin'));
const points=[...new Set(fg.index.array)].map(i=>new Vector3().fromBufferAttribute(fg.attributes.position,i));
for(let i=0;i<fg.index.count;i+=3){let p=new Vector3();for(let j=0;j<3;j++)p.add(new Vector3().fromBufferAttribute(fg.attributes.position,fg.index.array[i+j]));points.push(p.divideScalar(3));}

const percentile=(a,t)=>{if(!a.length)return null;a.sort((x,y)=>x-y);return a[Math.floor((a.length-1)*t)]};
const fields=points.filter((_,i)=>i%500===0).map(q=>({q,p:projectionPath(s,q)})).filter(x=>x.p),b=basis(s.projector),report=[];
for(const margin of[0,1,2,4,8])for(const focus of[250,300,350,400]){
 const state=structuredClone(s);state.foldMirror.width+=2*margin;state.foldMirror.height+=2*margin;
 const trace=createStrayTracer(state,fullHead,fullFace,kit);let kinds={},ghost=0,throughput=[],rms=[],zeroTargets=[];
 for(const {q,p}of fields){const chief=rayForUV(state,p.u,p.v),ref=trace(p.u,p.v),target=chief.at(focus/chief.direction.dot(b.f),new Vector3());let correct=0,dist2=0;
 for(let k=0;k<32;k++){const radius=4*Math.sqrt((k+.5)/32),angle=k*2.399963229728653,origin=b.p.clone().addScaledVector(b.r,radius*Math.cos(angle)).addScaledVector(b.u,radius*Math.sin(angle)),out=trace(p.u,p.v,8,new Ray(origin,target.clone().sub(origin).normalize()));kinds[out.kind]=(kinds[out.kind]||0)+1;if(out.kind==='face'){if(out.route.join('>')==='foldMirror>mirror'){correct++;if(ref.kind==='face')dist2+=out.point.distanceToSquared(ref.point);}else ghost++;}}
 throughput.push(correct/32);if(correct&&ref.kind==='face')rms.push(Math.sqrt(dist2/correct));if(!correct)zeroTargets.push(q.toArray());
 }
 const r={foldMarginPerSideMM:margin,assumedPupilDiameterMM:8,assumedFocusPlaneMM:focus,fields:fields.length,pupilSamples:32,zero:zeroTargets.length,zeroTargets,ghost,kinds,throughput:{min:Math.min(...throughput),p05:percentile(throughput,.05),median:percentile(throughput,.5)},rmsDisplacementFromChiefInnerHitMM:{p50:percentile(rms,.5),p95:percentile(rms,.95),max:percentile(rms,1)}};report.push(r);console.log({...r,zeroTargets:undefined});fs.writeFileSync('work/torso-flat/fold-focus-sweep.json',JSON.stringify(report,null,2));
}
