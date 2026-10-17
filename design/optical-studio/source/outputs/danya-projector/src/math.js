export const clone = x => JSON.parse(JSON.stringify(x));
export const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
export function defaults(){return {
 version:1,outputRotation:0,face:{scale:1,x:0,y:0},source:{rotation:270,flipX:false,flipY:false,crop:[.05,.12,.9,.82]},
 features:[
  ['eyeL','画面左の目',.273,.412,.09,.04],['eyeR','画面右の目',.715,.412,.09,.04],
  ['browL','画面左の眉',.275,.338,.115,.022],['browR','画面右の眉',.72,.338,.115,.022],
  ['nose','鼻',.497,.575,.075,.065],['mouth','口',.5,.743,.16,.052]
 ].map(([id,name,x,y,rx,ry])=>({id,name,x,y,rx,ry,dx:0,dy:0,sx:1,sy:1,angle:0})),pins:[],
 color:{exposure:0,contrast:1,gamma:1,saturation:1,warmth:0,tint:0,skin:'#d5a78c',skinMix:0,skinOnly:true},
 calibration:{mode:'manual',corners:[[.28,.04],[.72,.04],[.72,.96],[.28,.96]]},
 mask:{enabled:false,rx:.49,ry:.50,feather:.025},pattern:'off',blackout:false,gridSize:257,resolution:[1280,720]
};}
export function solve(A,b){
 const n=b.length,m=A.map((r,i)=>[...r,b[i]]);
 for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(m[i][k])>Math.abs(m[p][k]))p=i;
  if(Math.abs(m[p][k])<1e-12)throw Error('基準点が重なっています。少し離してください。');
  [m[k],m[p]]=[m[p],m[k]];const d=m[k][k];for(let j=k;j<=n;j++)m[k][j]/=d;
  for(let i=k+1;i<n;i++){const q=m[i][k];for(let j=k;j<=n;j++)m[i][j]-=q*m[k][j];}
 }
 const x=new Array(n);for(let i=n-1;i>=0;i--){x[i]=m[i][n];for(let j=i+1;j<n;j++)x[i]-=m[i][j]*x[j];}return x;
}
const radial = r2 => r2<1e-14?0:r2*Math.log(r2);
export function anchors(state){
 const src=[],dst=[];const add=(s,t)=>{src.push(s);dst.push(t);};
 for(const f of (state.deformFeatures?state.features:[])){const a=f.angle*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
  for(const [u,v]of [[0,0],[f.rx,0],[-f.rx,0],[0,f.ry],[0,-f.ry]])
   add([f.x+u,f.y+v],[f.x+f.dx+c*u*f.sx-s*v*f.sy,f.y+f.dy+s*u*f.sx+c*v*f.sy]);
 }
 for(const p of state.pins)add([p.x,p.y],[p.x+p.dx,p.y+p.dy]);
 for(const p of [[0,0],[.25,0],[.5,0],[.75,0],[1,0],[1,.25],[1,.5],[1,.75],[1,1],[.75,1],[.5,1],[.25,1],[0,1],[0,.75],[0,.5],[0,.25]])add(p,p);
 return {src,dst};
}
export function fitTPS(from,to){
 const n=from.length,N=n+3,A=Array.from({length:N},()=>Array(N).fill(0));
 for(let i=0;i<n;i++){for(let j=0;j<n;j++)A[i][j]=radial((from[i][0]-from[j][0])**2+(from[i][1]-from[j][1])**2);
  A[i][i]+=1e-8;A[i][n]=A[n][i]=1;A[i][n+1]=A[n+1][i]=from[i][0];A[i][n+2]=A[n+2][i]=from[i][1];}
 return {points:from,wx:solve(A,[...to.map(p=>p[0]),0,0,0]),wy:solve(A,[...to.map(p=>p[1]),0,0,0])};
}
export function evaluateTPS(t,x,y){const n=t.points.length;let u=t.wx[n]+t.wx[n+1]*x+t.wx[n+2]*y,v=t.wy[n]+t.wy[n+1]*x+t.wy[n+2]*y;
 for(let i=0;i<n;i++){const p=t.points[i],r=radial((x-p[0])**2+(y-p[1])**2);u+=r*t.wx[i];v+=r*t.wy[i];}return [u,v];}
export function warpField(state,size=state.gridSize){
 const {src,dst}=anchors(state),t=fitTPS(dst,src),data=new Float32Array(size*size*2);let minJ=Infinity;
 for(let j=0;j<size;j++)for(let i=0;i<size;i++){const p=evaluateTPS(t,i/(size-1),j/(size-1));data.set(p,2*(j*size+i));}
 for(let j=0;j<size-1;j++)for(let i=0;i<size-1;i++){const k=2*(j*size+i),dx=data[k+2]-data[k],dy=data[k+3]-data[k+1],ex=data[k+size*2]-data[k],ey=data[k+size*2+1]-data[k+1];minJ=Math.min(minJ,(dx*ey-dy*ex)*(size-1)**2);}
 if(!Number.isFinite(minJ)||minJ<.025)throw Error('変形が折り返すため未適用です。移動量・拡大率を戻してください。');
 return {data,size,minJ};
}
export function homography(corners){
 const target=[[0,0],[1,0],[1,1],[0,1]],A=[],b=[];
 for(let i=0;i<4;i++){const [x,y]=corners[i],[u,v]=target[i];A.push([x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]);b.push(u,v);}
 const h=solve(A,b);return [h[0],h[3],h[6],h[1],h[4],h[7],h[2],h[5],1];
}
export function validQuad(q){if(!Array.isArray(q)||q.length!==4)return false;let sign=0;
 for(let i=0;i<4;i++){const a=q[i],b=q[(i+1)%4],c=q[(i+2)%4];if(!a.every(Number.isFinite)||a.some(x=>Math.abs(x)>2))return false;const z=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]);if(Math.abs(z)<.005||sign&&Math.sign(z)!==sign)return false;sign=Math.sign(z);}return true;}
export function validateState(s){
 if(s?.version!==1||s.features?.length!==6||!Array.isArray(s.pins)||s.pins.length>8)throw Error('この設定ファイルの形式に対応していません。');
 s=clone(s);s.outputRotation??=0;s.face??={scale:1,x:0,y:0};
 const finite=(v,a,b)=>typeof v==='number'&&Number.isFinite(v)&&v>=a&&v<=b;
 if(![0,90,180,270].includes(s.outputRotation))throw Error('画面の回転角が不正です。');
 if(!finite(s.face.scale,.25,2.5)||!finite(s.face.x,-.5,.5)||!finite(s.face.y,-.5,.5))throw Error('顔全体の位置・大きさが範囲外です。');
 for(const f of [...s.features,...s.pins]){for(const k of ['x','y'])if(!finite(f[k],.01,.99))throw Error('基準点の位置が範囲外です。');for(const k of ['dx','dy'])if(!finite(f[k],-.3,.3))throw Error('移動量が範囲外です。');}
 for(const f of s.features){if(!finite(f.rx,.005,.2)||!finite(f.ry,.005,.2)||!finite(f.sx,.4,2)||!finite(f.sy,.4,2)||!finite(f.angle,-45,45))throw Error('部位の形状設定が範囲外です。');}
 if(!validQuad(s.calibration?.corners)||!['manual','v5'].includes(s.calibration.mode))throw Error('投影の四隅が交差しています。');
 if(![129,257,513].includes(s.gridSize)||![[1280,720],[1920,1080]].some(r=>String(r)===String(s.resolution)))throw Error('出力解像度が不正です。');
 if(![0,90,180,270].includes(s.source?.rotation)||!Array.isArray(s.source.crop)||s.source.crop.length!==4||!s.source.crop.every(v=>finite(v,0,1))||s.source.crop[2]<.1||s.source.crop[3]<.1||s.source.crop[0]+s.source.crop[2]>1.00001||s.source.crop[1]+s.source.crop[3]>1.00001)throw Error('切り抜き範囲が不正です。');
 const ranges={exposure:[-2,2],contrast:[.5,1.8],gamma:[.5,2],saturation:[0,2],warmth:[-.3,.3],tint:[-.2,.2],skinMix:[0,1]};
 for(const [k,r]of Object.entries(ranges))if(!finite(s.color?.[k],...r))throw Error('色調整の値が範囲外です。');
 if(!/^#[0-9a-f]{6}$/i.test(s.color.skin)||!finite(s.mask?.rx,.2,.7)||!finite(s.mask.ry,.2,.7)||!finite(s.mask.feather,.001,.15)||!['off','grid','white'].includes(s.pattern))throw Error('マスク・色の設定が不正です。');
 return clone(s);
}

export function rotateScreen(p,degrees,res,inverse=false){const a=degrees*Math.PI/180*(inverse?-1:1),c=Math.cos(a),s=Math.sin(a),x=(p[0]-.5)*res[0],y=(p[1]-.5)*res[1];return [.5+(c*x-s*y)/res[0],.5+(s*x+c*y)/res[1]];}
