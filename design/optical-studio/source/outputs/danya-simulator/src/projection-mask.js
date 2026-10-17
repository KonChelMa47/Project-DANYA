import {createStrayTracer} from './stray-light.js';
export const maskSignature=s=>JSON.stringify([s.design,s.modelScale,s.projector,s.preMirror,s.foldMirror,s.mirror,s.optics.throwRatio,s.optics.offset,s.apertureBlade||null]);
export const expectedRoute=s=>['preMirror','foldMirror','mirror'].filter(k=>s[k]).join('>');
export function maskAt(mask,u,v){const x=Math.floor(u*1280),y=Math.floor(v*720);return x>=0&&x<1280&&y>=0&&y<720&&mask[y*1280+x]===255;}
export async function buildProjectionMask(s,h,f,body,bounds,progress){
 const trace=createStrayTracer(s,h,f,body),expected=expectedRoute(s),mask=new Uint8Array(1280*720),offsets=[.001,.5,.999];
 const lo=bounds.min,hi=bounds.max;if(lo[0]>hi[0]||lo[1]>hi[1])return mask;
 const x0=Math.max(0,Math.floor(lo[0]*1280)-2),x1=Math.min(1279,Math.ceil(hi[0]*1280)+2),y0=Math.max(0,Math.floor(lo[1]*720)-2),y1=Math.min(719,Math.ceil(hi[1]*720)+2);
 for(let y=y0;y<=y1;y++){for(let x=x0;x<=x1;x++){let safe=true;outer:for(const dy of offsets)for(const dx of offsets){const result=trace((x+dx)/1280,(y+dy)/720);if(result.kind==='bounce-limit'||result.kind==='face'&&result.route.join('>')!==expected){safe=false;break outer;}}if(safe)mask[y*1280+x]=255;}if(y%4===0&&!await progress((y-y0+1)/(y1-y0+1)))return null;}return mask;
}
