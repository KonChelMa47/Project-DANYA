import {Vector3,Ray} from 'three';
import {basis,reflect,rayForUV,mirrorContains} from './optics.js';
export function outgoing(state,u,v,{pupilDiameter=0,pupilAxial=0,pupilX=0,pupilY=0,focus=350,mirrorType='front',thickness=1.5,index=1.52,unlimited=false}={}){
 const p=basis(state.projector),m=basis(state.foldMirror),chief=rayForUV(state,u,v).direction;
 const target=p.p.clone().addScaledVector(chief,focus/chief.dot(p.f));
 const origin=p.p.clone().addScaledVector(p.f,pupilAxial).addScaledVector(p.r,pupilX*pupilDiameter/2).addScaledVector(p.u,pupilY*pupilDiameter/2);
 const d=target.sub(origin).normalize(),den=d.dot(m.f);if(den>=-1e-9)return null;
 const t=m.p.clone().sub(origin).dot(m.f)/den;if(t<=0)return null;
 const hit=origin.clone().addScaledVector(d,t),local=hit.clone().sub(m.p),x=local.dot(m.r),y=local.dot(m.u);
 const inside=mirrorContains(state.foldMirror,x,y);
 const ideal=reflect(d,m.f),ghost={origin:hit,direction:ideal};let main=ghost,backInside=true;
 if(mirrorType==='back'){
  const cos=-den,st2=(1-cos*cos)/(index*index),ct=Math.sqrt(1-st2);
  const dt=d.clone().addScaledVector(m.f,cos).divideScalar(index).addScaledVector(m.f,-ct);
  const back=hit.clone().addScaledVector(dt,thickness/ct),dr=reflect(dt,m.f),exit=back.clone().addScaledVector(dr,thickness/ct);
  const b=back.clone().sub(m.p),e=exit.clone().sub(m.p);backInside=[b,e].every(q=>mirrorContains(state.foldMirror,q.dot(m.r),q.dot(m.u)));
  main={origin:exit,direction:ideal};
 }
 return{hit,x,y,inside:unlimited||inside,mainInside:unlimited||(inside&&backInside),main,ghost:mirrorType==='back'?ghost:null,incidenceDegrees:Math.acos(-den)*180/Math.PI};
}
export function ceiling(ray,z){if(ray.direction.z<=1e-9)return null;const t=(z-ray.origin.z)/ray.direction.z;return t>0?ray.origin.clone().addScaledVector(ray.direction,t):null;}
export function pupilSamples(n){return n<=1?[[0,0]]:Array.from({length:n},(_,i)=>{let a=i*2.399963229728653,r=Math.sqrt((i+.5)/n);return[r*Math.cos(a),r*Math.sin(a)];});}
export function mirrorVariant(state,width,height,dx=0,dy=0){const s=structuredClone(state),b=basis(s.foldMirror);s.foldMirror.width=width;s.foldMirror.height=height;s.foldMirror.position=b.p.addScaledVector(b.r,dx).addScaledVector(b.u,dy).toArray();return s;}
export function ceilingAligned(state){const s=structuredClone(state),p=basis(s.projector),m=basis(s.foldMirror),n=new Vector3(0,0,1).sub(p.f).normalize(),r=m.r.clone().addScaledVector(n,-m.r.dot(n)).normalize(),up=new Vector3().crossVectors(n,r);return {state:s,normal:n,right:r,up};}
