import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
import {MeshBVH} from 'three-mesh-bvh';
import {SPEC,vec,basis,quaternion,degrees,defaultState,aimProjector,projectionPath,visiblePath,traceRay,moveAlongLine,validateState,rayForUV} from './optics.js';

const $=id=>document.getElementById(id), pause=()=>new Promise(r=>setTimeout(r,0));
document.querySelectorAll('button,input,select').forEach(e=>e.disabled=true);
let state=defaultState(), selected='projector', geometry={}, originals={}, headBVH,faceBVH;
let surfacePoints=[];
let revision=0,timer,ready=false,activeCalculation=false,latestMetrics={},sourceOK=false;
const viewport=$('viewport'),scene=new THREE.Scene();
scene.background=new THREE.Color('#1b2428');
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.localClippingEnabled=true;
renderer.outputColorSpace=THREE.SRGBColorSpace;viewport.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(36,1,.5,5000);camera.up.set(0,0,1);
const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.target.set(0,0,130);
scene.add(new THREE.HemisphereLight(0xe8f3f5,0x3d4945,2.1));
const light=new THREE.DirectionalLight(0xffffff,2.6);light.position.set(220,-200,400);scene.add(light);
const fill=new THREE.DirectionalLight(0x91b8b4,1.3);fill.position.set(-200,100,220);scene.add(fill);
const grid=new THREE.GridHelper(650,26,0x425653,0x2b3b3a);grid.rotation.x=Math.PI/2;grid.position.z=-38;scene.add(grid);
const axes=new THREE.AxesHelper(45);axes.position.set(-110,-110,-37);scene.add(axes);
const projector=new THREE.Group(),mirror=new THREE.Group();scene.add(projector,mirror);
const cyan=new THREE.MeshStandardMaterial({color:0x52979e,metalness:.35,roughness:.5});
const body=new THREE.Mesh(new THREE.BoxGeometry(56.97,21,80.35),cyan);body.position.set(0,-7,-42);projector.add(body);
const lens=new THREE.Mesh(new THREE.CylinderGeometry(8,9,10,32),new THREE.MeshStandardMaterial({color:0xaff7f7,metalness:.7,roughness:.15,emissive:0x225459}));lens.rotation.x=Math.PI/2;lens.position.z=-1;projector.add(lens);
const lensAxis=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),34,0x7edfe6,5,3);projector.add(lensAxis);
const mirrorSurface=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshStandardMaterial({color:0xd0e0df,metalness:.7,roughness:.14,side:THREE.DoubleSide}));mirror.add(mirrorSurface);
const mirrorBack=new THREE.Mesh(new THREE.BoxGeometry(1,1,1.5),new THREE.MeshStandardMaterial({color:0x756144,metalness:.4,roughness:.6}));mirrorBack.position.z=-.85;mirror.add(mirrorBack);
const mirrorOutline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1,1)),new THREE.LineBasicMaterial({color:0xebbe77}));mirrorOutline.position.z=.1;mirror.add(mirrorOutline);
const normalArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),28,0xebbe77,5,3);mirror.add(normalArrow);
let headMesh,faceMesh,rayLines;
const transform=new TransformControls(camera,renderer.domElement);transform.setSize(.75);transform.setSpace('world');scene.add(transform.getHelper());
transform.addEventListener('dragging-changed',e=>{orbit.enabled=!e.value;if(!e.value)resetRail();});
transform.addEventListener('objectChange',()=>{
  const object=selected==='projector'?projector:mirror;
  state[selected].position=object.position.toArray();state[selected].rotation=degrees(object.quaternion);
  syncInputs();requestCalculation();
});
const sourceCanvas=$('source-preview'),ctx=sourceCanvas.getContext('2d'),rawCanvas=document.createElement('canvas');rawCanvas.width=1280;rawCanvas.height=720;
const raw=rawCanvas.getContext('2d'),texture=new THREE.CanvasTexture(sourceCanvas);texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;
const video=document.createElement('video');video.loop=true;video.muted=true;video.playsInline=true;let videoURL,sourceFrame=0,lastSourcePaint=0;
const uniforms={map:{value:texture},strength:{value:1}};
const faceMaterial=new THREE.ShaderMaterial({side:THREE.DoubleSide,uniforms,
vertexShader:`attribute vec2 projectionUV;attribute float illuminated;varying vec2 vUV;varying float vLight;varying vec3 vNormal;void main(){vUV=projectionUV;vLight=illuminated;vNormal=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
fragmentShader:`uniform sampler2D map;uniform float strength;varying vec2 vUV;varying float vLight;varying vec3 vNormal;void main(){float mask=smoothstep(.35,.75,vLight);vec3 base=vec3(.095,.12,.12)*(0.65+0.35*abs(normalize(vNormal).z));vec3 projected=texture2D(map,clamp(vUV,0.,1.)).rgb;gl_FragColor=vec4(base+projected*mask*strength,1.);#include <colorspace_fragment>}`.replace(';#include',';\n#include')
});
function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>$('toast').classList.remove('show'),4500);}
function updateObjects(){
  projector.position.fromArray(state.projector.position);projector.quaternion.copy(quaternion(state.projector.rotation));
  mirror.position.fromArray(state.mirror.position);mirror.quaternion.copy(quaternion(state.mirror.rotation));
  for(const o of [mirrorSurface,mirrorBack,mirrorOutline]) o.scale.set(state.mirror.width,state.mirror.height,1);
  grid.position.z=Math.min(-38,new THREE.Box3().setFromObject(projector).min.z-4);axes.position.z=grid.position.z+.1;
  uniforms.strength.value=state.display.exposure*state.optics.transmission*state.optics.reflectance;
  if(rayLines)rayLines.visible=state.display.rays;
  transform.attach(selected==='projector'?projector:mirror);
}
for(const [container,key,labels] of [['position-fields','position',['X · 前後 mm','Y · 左右 mm','Z · 上下 mm']],['rotation-fields','rotation',['RX °','RY °','RZ °']]]) {
  labels.forEach((label,i)=>{const l=document.createElement('label'),span=document.createElement('span'),input=document.createElement('input');span.textContent=label;input.type='number';input.id=`${key}-${i}`;input.step=key==='position'?'1':'1';input.min=key==='position'?'-3000':'-360';input.max=key==='position'?'3000':'360';input.addEventListener('change',()=>{if(!input.validity.valid||!Number.isFinite(input.valueAsNumber)){syncInputs();return;}state[selected][key][i]=input.valueAsNumber;updateObjects();resetRail();requestCalculation();});l.append(span,input);$(container).appendChild(l);});
}
function syncInputs(){
  for(const key of ['position','rotation'])state[selected][key].forEach((v,i)=>$(`${key}-${i}`).value=(Math.round(v*100)/100));
  $('part-title').textContent=selected==='projector'?'レンズ中心の位置':'反射面中心の位置';
  $('projector-controls').hidden=selected!=='projector';$('mirror-controls').hidden=selected!=='mirror';
  for(const [id,value] of [['mirror-width',state.mirror.width],['mirror-height',state.mirror.height],['aim-roll',state.aimRoll],['throw',state.optics.throwRatio],['offset',state.optics.offset],['reflectance',state.optics.reflectance],['transmission',state.optics.transmission],['model-scale',state.modelScale],['exposure',state.display.exposure]])$(id).value=value;
  $('source').value=state.source.kind;$('source-rotation').value=state.source.rotation;$('flip-x').checked=state.source.flipX;$('flip-y').checked=state.source.flipY;$('rays').checked=state.display.rays;
  $('video-upload').hidden=state.source.kind!=='video';$('video-controls').hidden=state.source.kind!=='video';
  document.querySelectorAll('[data-part]').forEach(b=>b.classList.toggle('active',b.dataset.part===selected));
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===state.display.view));
}
let railOrigin=state.projector.position.slice(),railAxis=[0,0,1];
function resetRail(){railOrigin=state[selected].position.slice();railAxis=$('axis').value==='local'?basis(state[selected]).f.toArray():({x:[1,0,0],y:[0,1,0],z:[0,0,1]})[$('axis').value];$('rail').value=0;$('rail-value').value='0 mm';}
$('rail').addEventListener('input',()=>{const d=+$('rail').value;state[selected].position=moveAlongLine(railOrigin,railAxis,d);$('rail-value').value=`${d} mm`;updateObjects();syncInputs();requestCalculation();});
$('axis').onchange=resetRail;$('rail-reset').onclick=resetRail;
document.querySelectorAll('[data-part]').forEach(b=>b.onclick=()=>{selected=b.dataset.part;updateObjects();syncInputs();resetRail();});
for(const mode of ['translate','rotate'])$(mode).onclick=()=>{transform.setMode(mode);$('translate').classList.toggle('active',mode==='translate');$('rotate').classList.toggle('active',mode==='rotate');};
$('aim').onclick=()=>{if(vec(state.projector.position).distanceTo(vec(state.mirror.position))<1){toast('レンズとミラーを離してください');return;}aimProjector(state);updateObjects();syncInputs();resetRail();requestCalculation();};
$('mirror-aim').onclick=()=>{
  const m=vec(state.mirror.position),incoming=m.clone().sub(vec(state.projector.position)).normalize(),outgoing=new THREE.Vector3(65,0,145).multiplyScalar(state.modelScale).sub(m).normalize();
  const n=outgoing.clone().sub(incoming).normalize(),old=basis(state.mirror).r,r=old.addScaledVector(n,-old.dot(n)).normalize();
  if(n.lengthSq()<.1||r.lengthSq()<.1){toast('この配置では中心反射面を決められません');return;}
  const u=new THREE.Vector3().crossVectors(n,r);state.mirror.rotation=degrees(new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(r,u,n)));
  aimProjector(state);
  updateObjects();syncInputs();resetRail();requestCalculation();
};
function bindNumber(id,object,key,calc=true){$(id).onchange=()=>{const input=$(id);if(!input.validity.valid||!Number.isFinite(input.valueAsNumber)){syncInputs();return;}object()[key]=input.valueAsNumber;updateObjects();if(calc)requestCalculation();};}
bindNumber('mirror-width',()=>state.mirror,'width');bindNumber('mirror-height',()=>state.mirror,'height');bindNumber('aim-roll',()=>state,'aimRoll',false);
for(const [id,key]of [['throw','throwRatio'],['offset','offset'],['reflectance','reflectance'],['transmission','transmission']])bindNumber(id,()=>state.optics,key,id==='throw'||id==='offset');
$('model-scale').onchange=()=>{if(!$('model-scale').validity.valid||!Number.isFinite($('model-scale').valueAsNumber)){syncInputs();return;}state.modelScale=+$('model-scale').value;rebuildGeometry();requestCalculation();};
$('exposure').oninput=()=>{state.display.exposure=+$('exposure').value;updateObjects();};
$('rays').onchange=()=>{state.display.rays=$('rays').checked;updateObjects();};
function view(kind){state.display.view=kind;headMesh.material.clippingPlanes=kind==='inside'?[new THREE.Plane(new THREE.Vector3(0,1,0),0)]:[];headMesh.material.needsUpdate=true;
  $('view-caption').textContent=kind==='inside'?'頭部の片側を断面表示 · 遮蔽計算は完全なシェル':'不透明な頭部と、拡散透過する顔';
  if(kind==='inside'){camera.position.set(380,-470,260);orbit.target.set(0,0,132);}syncInputs();}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{if(ready)view(b.dataset.view);});
function cameraView(which){const s=state.modelScale;orbit.target.set(0,0,130*s);camera.position.copy(vec(({front:[620,0,145],side:[140,-650,190],fit:[485,-380,265]})[which]).multiplyScalar(s));orbit.update();}
for(const id of ['front','side','fit'])$(id).onclick=()=>cameraView(id);
function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('save').onclick=()=>download(new Blob([JSON.stringify(state,null,2)],{type:'application/json'}),'DANYA-optical-layout.json');
$('load').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const next=validateState(JSON.parse(await file.text())),scaleChanged=state.modelScale!==next.modelScale;state=next;if(scaleChanged)rebuildGeometry();updateObjects();syncInputs();resetRail();view(state.display.view);sourceChanged();requestCalculation();toast('配置を読み込みました');}catch(error){toast(error.message);}e.target.value='';};
$('reset').onclick=()=>{const scaled=state.modelScale!==1;state=defaultState();selected='projector';if(scaled)rebuildGeometry();updateObjects();syncInputs();resetRail();view('outside');cameraView('fit');sourceChanged();requestCalculation();toast('初期配置に戻しました');};
$('recommended').onclick=async()=>{try{const response=await fetch('recommended-layout.json',{cache:'no-store'});if(!response.ok)throw Error('推奨配置を準備中です');const next=validateState(await response.json()),scaleChanged=state.modelScale!==next.modelScale;state=next;selected='projector';if(scaleChanged)rebuildGeometry();updateObjects();syncInputs();resetRail();view('outside');cameraView('fit');sourceChanged();requestCalculation();toast('後頭部側の低い位置に配置しました');}catch(e){toast(e.message);}};
$('capture').onclick=()=>renderer.domElement.toBlob(b=>download(b,'DANYA-simulation.png'));
function pattern(){raw.fillStyle='#243737';raw.fillRect(0,0,1280,720);for(let y=0;y<9;y++)for(let x=0;x<16;x++){raw.fillStyle=(x+y)%2?'#aecaba':'#527a6d';raw.fillRect(x*80+1,y*80+1,78,78);}raw.strokeStyle='#fff';raw.lineWidth=4;raw.beginPath();raw.moveTo(640,0);raw.lineTo(640,720);raw.moveTo(0,360);raw.lineTo(1280,360);raw.stroke();raw.fillStyle='#0b201c';raw.fillRect(335,255,610,210);raw.fillStyle='#e5f5ea';raw.textAlign='center';raw.font='bold 56px sans-serif';raw.fillText('DANYA  ↑ TOP',640,330);raw.font='32px sans-serif';raw.fillText('L   1280 × 720   R',640,408);}
function paintSource(){
  ctx.fillStyle='#000';ctx.fillRect(0,0,1280,720);ctx.save();ctx.translate(640,360);ctx.rotate(state.source.rotation*Math.PI/180);ctx.scale(state.source.flipX?-1:1,state.source.flipY?-1:1);
  // Keep the entire frame when rotating; never silently crop the source.
  const scale=state.source.rotation%180===0?1:720/1280;ctx.scale(scale,scale);ctx.drawImage(rawCanvas,-640,-360);ctx.restore();texture.needsUpdate=true;
}
function sourceChanged(){
  if(state.source.kind==='pattern'){pattern();paintSource();$('source-status').textContent='テストパターン';$('source-status').className='badge';}
  if(state.source.kind==='video'&&!videoURL){raw.fillStyle='#0d1717';raw.fillRect(0,0,1280,720);raw.fillStyle='#9bb6aa';raw.font='32px sans-serif';raw.textAlign='center';raw.fillText('動画を選択してください',640,360);paintSource();$('source-status').textContent='動画未選択';$('source-status').className='badge';}
  if(state.source.kind!=='video')video.pause();else if(videoURL)video.play().catch(()=>{});
  if(state.source.kind==='live'){sourceFrame=-1;$('source-status').textContent='接続中';}
  syncInputs();
}
$('source').onchange=()=>{state.source.kind=$('source').value;sourceChanged();};
$('source-rotation').onchange=()=>{state.source.rotation=+$('source-rotation').value;paintSource();};
for(const [id,key]of [['flip-x','flipX'],['flip-y','flipY']])$(id).onchange=()=>{state.source[key]=$(id).checked;paintSource();};
$('video-file').onchange=e=>{const file=e.target.files[0];if(!file)return;if(videoURL)URL.revokeObjectURL(videoURL);videoURL=URL.createObjectURL(file);video.src=videoURL;state.source.kind='video';sourceChanged();video.play().catch(()=>toast('再生ボタンを押してください'));};
$('video-play').onclick=()=>video.paused?video.play():video.pause();video.onerror=()=>toast('この動画形式はブラウザーで再生できません');
async function pollSource(){
  try{if(state.source.kind==='live'){
    const status=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());sourceOK=status.live;
    $('source-status').textContent=status.live?'LIVE · オーキャン v2':'v2 停止 / 接続待ち';$('source-status').className=`badge ${status.live?'live':'offline'}`;
    if(status.frame!==sourceFrame&&status.frame>0){const response=await fetch('/api/frame',{cache:'no-store'});if(!response.ok)throw Error('feed');const bitmap=await createImageBitmap(await response.blob());if(state.source.kind==='live'){raw.drawImage(bitmap,0,0,1280,720);paintSource();sourceFrame=status.frame;}bitmap.close();}
    if(!status.live&&sourceFrame<=0){raw.fillStyle='#101b1c';raw.fillRect(0,0,1280,720);raw.fillStyle='#96b4a6';raw.font='30px sans-serif';raw.textAlign='center';raw.fillText('オーキャン v2 の起動を待っています',640,355);paintSource();}
  }}catch(e){$('source-status').textContent='ソースに接続できません';$('source-status').className='badge offline';}setTimeout(pollSource,120);
}
function rebuildGeometry(){
  revision++;
  for(const key of ['head','face']){
    const a=geometry[key].attributes.position.array,src=originals[key];for(let i=0;i<a.length;i++)a[i]=src[i]*state.modelScale;geometry[key].attributes.position.needsUpdate=true;geometry[key].computeBoundingSphere();
  }
  headBVH=new MeshBVH(geometry.head,{maxLeafSize:8});faceBVH=new MeshBVH(geometry.face,{maxLeafSize:8});
  surfacePoints=[];const scale=state.modelScale;for(let z=43.33;z<221;z+=2)for(let y=-64.66;y<65.23;y+=2){const ray=new THREE.Ray(new THREE.Vector3(400*scale,y*scale,z*scale),new THREE.Vector3(-1,0,0)),hit=faceBVH.raycastFirst(ray,THREE.DoubleSide,0,500*scale);if(!hit)continue;const hh=headBVH.raycastFirst(ray,THREE.DoubleSide,0,500*scale);if(!hh||hh.distance>=hit.distance-.1)surfacePoints.push(hit.point);}
}
async function loadGeometry(name){
  const buffers=await Promise.all(['positions','indices','normals'].map(async suffix=>{const r=await fetch(`assets/${name}.${suffix}.bin`);if(!r.ok)throw Error('形状ファイルが見つかりません');return r.arrayBuffer();}));
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buffers[0]),3));g.setIndex(new THREE.BufferAttribute(new Uint32Array(buffers[1]),1));g.setAttribute('normal',new THREE.BufferAttribute(new Float32Array(buffers[2]),3));originals[name]=g.attributes.position.array.slice();return g;
}
function requestCalculation(){if(!ready)return;const id=++revision;clearTimeout(timer);$('computation').textContent='配置変更 · 計算待ち';timer=setTimeout(()=>calculate(id),140);}
async function calculate(id){
  const snapshot=structuredClone(state),uv=new Float32Array(geometry.face.attributes.position.count*2),lit=new Float32Array(geometry.face.attributes.position.count),positions=geometry.face.attributes.position;
  const h=headBVH,f=faceBVH;activeCalculation=true;const start=performance.now();
  for(let i=0;i<positions.count;i++){
    const p=new THREE.Vector3().fromBufferAttribute(positions,i),path=projectionPath(snapshot,p);
    if(path){uv[2*i]=path.u;uv[2*i+1]=path.v;lit[i]=visiblePath(snapshot,p,h,f,path)?1:0;}
    if(i%2048===0){if(id!==revision)return;$('computation').textContent=`光路・遮蔽を計算 ${Math.round(i/positions.count*100)}%`;await pause();}
  }
  if(id!==revision)return;
  geometry.face.setAttribute('projectionUV',new THREE.BufferAttribute(uv,2));geometry.face.setAttribute('illuminated',new THREE.BufferAttribute(lit,1));
  const counts={face:0,blocked:0,'mirror-miss':0,escape:0},lengths=[],lines=[],colors=[];
  for(let y=0;y<23;y++)for(let x=0;x<41;x++){
    const ray=traceRay(snapshot,(x+.5)/41,(y+.5)/23,h,f);counts[ray.kind]++;if(ray.kind==='face')lengths.push(ray.distance);
    if(x%5===0&&y%4===0 || x===20&&y===11){const color=new THREE.Color(ray.kind==='face'?0xa7e5c4:ray.kind==='blocked'?0xe89572:0x587e83);for(let k=1;k<ray.points.length;k++){lines.push(...ray.points[k-1].toArray(),...ray.points[k].toArray());colors.push(...color.toArray(),...color.toArray());}}
  }
  if(rayLines){scene.remove(rayLines);rayLines.geometry.dispose();rayLines.material.dispose();}
  const rg=new THREE.BufferGeometry();rg.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));rg.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));rayLines=new THREE.LineSegments(rg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.42,depthWrite:false}));rayLines.visible=state.display.rays;scene.add(rayLines);
  const total=41*23,pct=n=>(100*n/total).toFixed(1),lo=lengths.length?Math.min(...lengths):0,hi=lengths.length?Math.max(...lengths):0;
  let surfaceLit=0;for(const point of surfacePoints){const path=projectionPath(snapshot,point);if(path&&visiblePath(snapshot,point,h,f,path))surfaceLit++;}
  const surfaceCoverage=100*surfaceLit/surfacePoints.length;
  $('coverage').innerHTML=`${surfaceCoverage.toFixed(1)}<small>%</small>`;
  $('surface-note').textContent=`正面2 mm格子 · 画像光線の到達率 ${pct(counts.face)}%`;
  const mirrorPlane=new THREE.PlaneGeometry(1,1),mirrorMatrix=new THREE.Matrix4().compose(vec(snapshot.mirror.position),quaternion(snapshot.mirror.rotation),new THREE.Vector3(snapshot.mirror.width,snapshot.mirror.height,1));
  const mirrorCollision=h.intersectsGeometry(mirrorPlane,mirrorMatrix)||f.intersectsGeometry(mirrorPlane,mirrorMatrix);mirrorPlane.dispose();$('fit-warning').hidden=!mirrorCollision;$('fit-warning').textContent='ミラーがシェルと干渉しています';$('mirror-miss').innerHTML=`${pct(counts['mirror-miss'])}<small>%</small>`;
  $('path-length').innerHTML=lengths.length?`${Math.round(lo)}–${Math.round(hi)}<small>mm</small>`:'—<small>mm</small>';
  $('blocked-rate').textContent=`遮蔽 ${pct(counts.blocked)}% · 反射後に外れる ${pct(counts.escape)}%`;
  $('focus-status').textContent=!lengths.length?'顔へ届く光線がありません':lo<SPEC.focusMin||hi>SPEC.focusMax?'公称フォーカス範囲外の光路を含みます':'総光路長は公称範囲内（合焦の保証ではありません）';
  $('computation').textContent=`計算完了 · ${((performance.now()-start)/1000).toFixed(1)} s`;
  latestMetrics={counts,lo,hi,surfaceCoverage,surfaceLit,surfaceSamples:surfacePoints.length,mirrorCollision,illuminatedVertices:lit.reduce((a,b)=>a+b,0),revision:id};activeCalculation=false;
  window.dispatchEvent(new CustomEvent('danya-calculated',{detail:latestMetrics}));
}
function resize(){const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe(viewport);
function animate(now){requestAnimationFrame(animate);orbit.update();if(state.source.kind==='video'&&video.readyState>=2&&now-lastSourcePaint>60){raw.fillStyle='#000';raw.fillRect(0,0,1280,720);const fit=Math.min(1280/video.videoWidth,720/video.videoHeight),w=video.videoWidth*fit,h=video.videoHeight*fit;raw.drawImage(video,(1280-w)/2,(720-h)/2,w,h);paintSource();lastSourcePaint=now;$('source-status').textContent=video.paused?'動画 · 一時停止':'動画 · 再生中';}renderer.render(scene,camera);}
async function init(){
  try{[geometry.head,geometry.face]=await Promise.all([loadGeometry('head'),loadGeometry('face')]);$('load-detail').textContent='835,636 三角形の遮蔽構造を構築中';await pause();
    rebuildGeometry();geometry.face.setAttribute('projectionUV',new THREE.BufferAttribute(new Float32Array(geometry.face.attributes.position.count*2),2));geometry.face.setAttribute('illuminated',new THREE.BufferAttribute(new Float32Array(geometry.face.attributes.position.count),1));
    headMesh=new THREE.Mesh(geometry.head,new THREE.MeshStandardMaterial({color:0xa8b3ad,roughness:.65,metalness:.07,side:THREE.DoubleSide}));faceMesh=new THREE.Mesh(geometry.face,faceMaterial);scene.add(headMesh,faceMesh);
    ready=true;document.querySelectorAll('button,input,select').forEach(e=>e.disabled=false);$('loading').hidden=true;cameraView('fit');updateObjects();syncInputs();resetRail();sourceChanged();pollSource();requestCalculation();
    // Read-only inspection hooks for reproducible browser verification.
    window.danya={get state(){return structuredClone(state)},get metrics(){return structuredClone(latestMetrics)},get ready(){return ready},get sourceLive(){return sourceOK},get calculating(){return activeCalculation},get renderer(){return renderer}};
  }catch(error){$('loading').querySelector('strong').textContent='起動できませんでした';$('load-detail').textContent=error.message;console.error(error);}
}
syncInputs();resize();requestAnimationFrame(animate);init();
