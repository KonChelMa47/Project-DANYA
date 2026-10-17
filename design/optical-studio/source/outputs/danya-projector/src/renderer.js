import {homography} from './math.js';
const VS=`#version 300 es
in vec2 position;out vec2 uv;void main(){gl_Position=vec4(position,0,1);uv=vec2(position.x*.5+.5,.5-position.y*.5);}`;
const FS=`#version 300 es
precision highp float;precision highp usampler2D;
in vec2 uv;out vec4 frag;
uniform sampler2D sourceImage,field;uniform usampler2D opticalMap;
uniform mat3 projectorInverse;uniform int mode,flatView,rawView,blackout,pattern,masked,skinOnly;
uniform vec2 outputSize,maskRadius;uniform float feather,exposure,contrastValue,gammaValue,saturationValue,warmth,tint,skinMix;
uniform float outputAngle;uniform vec3 skinColor,faceTransform;uniform vec4 features[6],moves[6];uniform float angles[6];
bool outside(vec2 p){return any(lessThan(p,vec2(0)))||any(greaterThan(p,vec2(1)));}
vec2 warp(vec2 p){vec2 z=p*vec2(textureSize(field,0)-1);ivec2 a=ivec2(floor(z)),m=textureSize(field,0)-1;vec2 f=fract(z);
return mix(mix(texelFetch(field,clamp(a,ivec2(0),m),0).rg,texelFetch(field,clamp(a+ivec2(1,0),ivec2(0),m),0).rg,f.x),mix(texelFetch(field,clamp(a+ivec2(0,1),ivec2(0),m),0).rg,texelFetch(field,clamp(a+1,ivec2(0),m),0).rg,f.x),f.y);}
vec3 optical(vec2 p){vec2 z=p*vec2(textureSize(opticalMap,0))-.5;ivec2 a=ivec2(floor(z)),m=textureSize(opticalMap,0)-1;vec2 f=fract(z);vec2 sum=vec2(0);float weight=0.;
for(int j=0;j<2;j++)for(int i=0;i<2;i++){uvec2 v=texelFetch(opticalMap,clamp(a+ivec2(i,j),ivec2(0),m),0).rg;float w=(i==0?1.-f.x:f.x)*(j==0?1.-f.y:f.y);if(v.x<65535u&&v.y<65535u){sum+=vec2(v)/65534.*w;weight+=w;}}
return vec3(sum/max(weight,1e-8),weight);}
// Rigid feature layers over a skin-filled source. The core is not stretched by neighbors.
bool moved(int i){return length(moves[i].xy)>.000001||length(moves[i].zw-1.)>.000001||abs(angles[i])>.000001;}
vec3 skinFill(vec2 p,int k){
 vec4 f=features[k];vec3 sum=vec3(0);float total=0.;
 for(int j=0;j<24;j++){float a=6.2831853*(float(j)+.5)/24.;vec2 q=f.xy+f.zw*1.65*vec2(cos(a),sin(a));float weight=1./max(dot((p-q)/f.zw,(p-q)/f.zw),.08);
 for(int n=0;n<6;n++){if(n!=k)weight*=smoothstep(1.,1.4,length((q-features[n].xy)/features[n].zw));}
 sum+=texture(sourceImage,clamp(q,vec2(0),vec2(1))).rgb*weight;total+=weight;
 }return sum/max(total,.00001);
}
vec3 composite(vec2 p){vec3 c=texture(sourceImage,p).rgb;
 for(int i=0;i<6;i++){if(!moved(i))continue;float d=length((p-features[i].xy)/features[i].zw);if(d<1.5)c=mix(c,skinFill(p,i),1.-smoothstep(1.05,1.5,d));}
 for(int i=0;i<6;i++){if(!moved(i))continue;vec4 f=features[i],m=moves[i];float a=angles[i],co=cos(a),si=sin(a);vec2 d=p-f.xy-m.xy;vec2 q=vec2(co*d.x+si*d.y,-si*d.x+co*d.y)/m.zw;float rad=length(q/f.zw);if(rad<1.5){vec2 original=f.xy+q;vec3 layerColor=texture(sourceImage,clamp(original,vec2(0),vec2(1))).rgb;float blend=1.-smoothstep(1.,1.5,rad);c=mix(c,layerColor,blend);}}
 return c;
}
void main(){if(blackout==1){frag=vec4(0,0,0,1);return;}vec2 p;float coverage=1.;vec2 pixel=(uv-.5)*outputSize;float co=cos(outputAngle),si=sin(outputAngle);vec2 screenUV=vec2(co*pixel.x+si*pixel.y,-si*pixel.x+co*pixel.y)/outputSize+.5;
if(flatView==1){float w=(720./1000.)/(outputSize.x/outputSize.y);p=vec2((screenUV.x-.5)/w+.5,screenUV.y);}
else{vec3 q=projectorInverse*vec3(screenUV,1);if(abs(q.z)<1e-7){frag=vec4(0,0,0,1);return;}p=q.xy/q.z;}
if(flatView==0&&mode==1&&outside(p)){frag=vec4(0,0,0,1);return;}
if(flatView==0&&mode==1){vec3 q=optical(p);p=q.xy;coverage=q.z;}
p=(p-.5-faceTransform.yz)/faceTransform.x+.5;if(outside(p)){frag=vec4(0,0,0,1);return;}
vec2 s=rawView==1?p:warp(p);if(outside(s)){frag=vec4(0,0,0,1);return;}
vec3 c=rawView==1?texture(sourceImage,s).rgb:composite(s);
if(pattern>0){vec2 d=abs(fract(s*20.-.5)-.5)/max(fwidth(s*20.),vec2(.001));float line=1.-smoothstep(.6,1.6,min(d.x,d.y));c=pattern==2?vec3(1):mix(vec3(.12),vec3(.8,.9,1),line);}
else if(rawView==0){float skinWeight=1.;if(skinOnly==1){float cb=.5+dot(c,vec3(-.168736,-.331264,.5)),cr=.5+dot(c,vec3(.5,-.418688,-.081312));skinWeight=smoothstep(.26,.34,cb)*(1.-smoothstep(.52,.58,cb))*smoothstep(.49,.54,cr)*(1.-smoothstep(.71,.77,cr));
for(int i=0;i<6;i++){if(i==4)continue;vec4 e=features[i];e.xy+=moves[i].xy;e.zw*=moves[i].zw;skinWeight*=smoothstep(.8,1.6,length((s-e.xy)/max(e.zw,vec2(.001))));}}
float luminance=dot(c,vec3(.2126,.7152,.0722));vec3 adjusted=mix(vec3(luminance),c,saturationValue);adjusted=(adjusted-.5)*contrastValue+.5;adjusted*=exp2(exposure);adjusted*=vec3(1.+warmth,1.+tint,1.-warmth);
vec3 tinted=adjusted*skinColor/max(dot(skinColor,vec3(.2126,.7152,.0722)),.01);adjusted=mix(adjusted,tinted,skinMix);
c=mix(c,pow(max(adjusted,vec3(0)),vec3(1./gammaValue)),skinOnly==1?skinWeight:1.);}
if(masked==1&&rawView==0)coverage*=1.-smoothstep(1.-feather,1.,length((p-.5)/maskRadius));frag=vec4(clamp(c,0.,1.)*coverage,1);}`;
export class Renderer{
 constructor(canvas){this.canvas=canvas;const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});if(!gl)throw Error('WebGL 2が必要です。ChromeまたはEdgeで開いてください。');this.gl=gl;
  const compile=(type,src)=>{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
  this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,VS));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,FS));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));gl.useProgram(this.program);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);const a=gl.getAttribLocation(this.program,'position');gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,2,gl.FLOAT,false,0,0);this.uniforms=new Map();
  this.textures=[0,1,2].map(i=>{gl.activeTexture(gl.TEXTURE0+i);const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,i===0?gl.LINEAR:gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,i===0?gl.LINEAR:gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t;});
  this.i('sourceImage',0);this.i('field',1);this.i('opticalMap',2);this.setField({size:2,data:new Float32Array([0,0,1,0,0,1,1,1])});
  gl.activeTexture(gl.TEXTURE2);gl.texImage2D(gl.TEXTURE_2D,0,gl.RG16UI,1,1,0,gl.RG_INTEGER,gl.UNSIGNED_SHORT,new Uint16Array([65535,65535]));this.opticalReady=false;
 }
 loc(n){if(!this.uniforms.has(n))this.uniforms.set(n,this.gl.getUniformLocation(this.program,n));return this.uniforms.get(n);}
 i(n,v){this.gl.uniform1i(this.loc(n),v);} f(n,v){this.gl.uniform1f(this.loc(n),v);}
 setField({size,data}){const g=this.gl;g.activeTexture(g.TEXTURE1);g.bindTexture(g.TEXTURE_2D,this.textures[1]);g.texImage2D(g.TEXTURE_2D,0,g.RG32F,size,size,0,g.RG,g.FLOAT,data);}
 upload(source){const g=this.gl;g.activeTexture(g.TEXTURE0);g.bindTexture(g.TEXTURE_2D,this.textures[0]);g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL,false);g.texImage2D(g.TEXTURE_2D,0,g.RGBA,g.RGBA,g.UNSIGNED_BYTE,source);}
 async loadOptical(){if(this.opticalReady)return;if(this.opticalPromise)return this.opticalPromise;this.opticalPromise=(async()=>{const r=await fetch('assets/head-v5.uv16');if(!r.ok)throw Error('v5補正データを読み込めません');const data=await r.arrayBuffer();if(data.byteLength!==1280*720*4)throw Error('補正データのサイズが不正です');const g=this.gl;g.activeTexture(g.TEXTURE2);g.bindTexture(g.TEXTURE_2D,this.textures[2]);g.texImage2D(g.TEXTURE_2D,0,g.RG16UI,1280,720,0,g.RG_INTEGER,g.UNSIGNED_SHORT,new Uint16Array(data));this.opticalReady=true;})();try{await this.opticalPromise;}finally{this.opticalPromise=null;}}
 draw(s,view='projection',raw=false){const g=this.gl,[w,h]=s.resolution;if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}g.viewport(0,0,w,h);g.useProgram(this.program);
  this.f('outputAngle',(s.outputRotation||0)*Math.PI/180);g.uniform3f(this.loc('faceTransform'),s.face?.scale??1,s.face?.x??0,s.face?.y??0);g.uniform4fv(this.loc('moves[0]'),s.features.flatMap(f=>[f.dx,f.dy,f.sx,f.sy]));g.uniform1fv(this.loc('angles[0]'),s.features.map(f=>f.angle*Math.PI/180));g.uniform2f(this.loc('outputSize'),w,h);g.uniformMatrix3fv(this.loc('projectorInverse'),false,homography(s.calibration.corners));this.i('flatView',view==='face'?1:0);this.i('rawView',raw?1:0);this.i('blackout',s.blackout||s.calibration.mode==='v5'&&!this.opticalReady&&view!=='face'?1:0);this.i('mode',s.calibration.mode==='v5'?1:0);this.i('pattern',['off','grid','white'].indexOf(s.pattern));this.i('masked',s.mask.enabled?1:0);this.i('skinOnly',s.color.skinOnly?1:0);
  g.uniform2f(this.loc('maskRadius'),s.mask.rx,s.mask.ry);this.f('feather',s.mask.feather);for(const [u,k]of Object.entries({exposure:'exposure',contrastValue:'contrast',gammaValue:'gamma',saturationValue:'saturation',warmth:'warmth',tint:'tint',skinMix:'skinMix'}))this.f(u,s.color[k]);const rgb=s.color.skin.match(/\w\w/g).map(n=>parseInt(n,16)/255);g.uniform3fv(this.loc('skinColor'),rgb);g.uniform4fv(this.loc('features[0]'),s.features.flatMap(f=>[f.x,f.y,f.rx,f.ry]));g.drawArrays(g.TRIANGLES,0,3);
 }
}
