import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
const base='assets/buildable-bust-v9/', view=document.getElementById('view');
const renderer=new T.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));view.prepend(renderer.domElement);
const scene=new T.Scene();scene.background=new T.Color(0x101b22);
scene.add(new T.HemisphereLight(0xe9ffff,0x34433e,2.5));const light=new T.DirectionalLight(0xffffff,3);light.position.set(80,200,170);scene.add(light);
const camera=new T.PerspectiveCamera(38,1,.1,2000);camera.up.set(0,0,1);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(67.5,10,37.5);controls.enableDamping=true;
const cameras={oblique:[250,230,190],top:[67.5,-300,37.6],side:[280,15,37.5]};
function viewpoint(id){camera.position.set(...cameras[id]);controls.update();}
Object.keys(cameras).forEach(id=>document.getElementById(id).onclick=()=>viewpoint(id));viewpoint('oblique');
const loader=new STLLoader(),fixings=[];let projector;
const state=document.getElementById('state');
async function get(url,kind='json'){const r=await fetch(url);if(!r.ok)throw new Error(`${r.status}: ${url}`);return kind==='json'?r.json():r.arrayBuffer();}
try{
  const [manifest,datum,pb,ib]=await Promise.all([get(base+'projector-tray-candidates.json'),get(base+'projector-tray-datum.json'),get('assets/ti-kit-rd/kit.positions.bin','bin'),get('assets/ti-kit-rd/kit.indices.bin','bin')]);
  const toRaw=new T.Matrix4().fromArray(datum.projectorMatrix).multiply(new T.Matrix4().makeTranslation(...datum.rawOrigin.map(v=>-v))).invert();
  await Promise.all(manifest.parts.map(async p=>{
    const g=await loader.loadAsync(base+p.name+'.stl');g.applyMatrix4(toRaw);
    const color=p.material.startsWith('printed')?0x319b84:p.material.includes('silicone')?0x30393b:0xb8c0c4;
    const m=new T.Mesh(g,new T.MeshStandardMaterial({color,metalness:p.material.startsWith('purchased')?.5:0,roughness:.55}));scene.add(m);
    if(/-(clip|pad|bolt|washer|nut)-/.test(p.name))fixings.push(m);
  }));
  const g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(new Float32Array(pb),3));g.setIndex(new T.BufferAttribute(new Uint32Array(ib),1));g.translate(...datum.rawOrigin);g.computeVertexNormals();
  projector=new T.Mesh(g,new T.MeshStandardMaterial({color:0xcea953,roughness:.62,side:T.DoubleSide}));scene.add(projector);
  function update(){const lift=Number(document.getElementById('lift').value);projector.position.y=lift;projector.visible=document.getElementById('projector').checked;
    const transparent=document.getElementById('transparent').checked;projector.material.transparent=transparent;projector.material.opacity=transparent?.22:1;projector.material.depthWrite=!transparent;
    fixings.forEach(m=>m.visible=lift===0);document.getElementById('lift-value').value=lift+' mm';state.textContent=lift?'押さえを取り外した状態の分解表示':'135 × 75 mmの底面を支持。底面ネジ穴は使用しません。';}
  ['projector','transparent','lift'].forEach(id=>document.getElementById(id).oninput=update);update();
}catch(e){state.textContent='モデルを読み込めませんでした：'+e.message;console.error(e);}
function resize(){renderer.setSize(view.clientWidth,view.clientHeight);camera.aspect=view.clientWidth/view.clientHeight;camera.updateProjectionMatrix();}new ResizeObserver(resize).observe(view);resize();
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
