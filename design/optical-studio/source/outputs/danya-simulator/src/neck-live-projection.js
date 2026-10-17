import * as T from 'three';
import {FaceWarp} from './face-warp.js';
// Geometric full-frame preview. This does not simulate radiometry or defocus.
export async function attachLiveProjection(mesh,state,base,preview,status){
 const get=async name=>{const r=await fetch(base+name);if(!r.ok)throw Error(name);return r.arrayBuffer();};
 const [uvData,litData,outerData,maskData,metadata]=await Promise.all([get('face.projection-uv.bin'),get('face.illuminated.bin'),get('face.outer.indices.bin'),get('experimental-mask.bin'),fetch(base+'projection-metadata.json').then(r=>r.json())]);
 if(JSON.stringify(metadata.layout)!==JSON.stringify(state)||metadata.vertices!==mesh.geometry.attributes.position.count)throw Error('配置と映像変換データが一致しません');
 const uv=new Float32Array(uvData),lit=new Float32Array(litData),mask=new Uint8Array(maskData),g=mesh.geometry;g.userData.outerIndices=new Uint32Array(outerData);g.setAttribute('projectionUV',new T.BufferAttribute(uv,2));g.setAttribute('illuminated',new T.BufferAttribute(lit,1));
 const warp=new FaceWarp();warp.update(g,uv,lit);preview.width=1280;preview.height=720;const c=preview.getContext('2d'),maskCanvas=document.createElement('canvas');maskCanvas.width=1280;maskCanvas.height=720;const mc=maskCanvas.getContext('2d'),im=mc.createImageData(1280,720);
 for(let y=0;y<720;y++)for(let x=0;x<1280;x++){const i=((719-y)*1280+x)*4;im.data[i]=im.data[i+1]=im.data[i+2]=mask[y*1280+x];im.data[i+3]=255;}mc.putImageData(im,0,0);
 const texture=new T.CanvasTexture(preview);texture.colorSpace=T.SRGBColorSpace;texture.minFilter=T.LinearFilter;texture.generateMipmaps=false;
 const material=new T.ShaderMaterial({side:T.DoubleSide,uniforms:{map:{value:texture}},vertexShader:'attribute vec2 projectionUV;attribute float illuminated;varying vec2 vUV;varying float vLight;void main(){vUV=projectionUV;vLight=illuminated;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',fragmentShader:'uniform sampler2D map;varying vec2 vUV;varying float vLight;void main(){vec3 color=texture2D(map,clamp(vUV,0.,1.)).rgb;gl_FragColor=vec4(vec3(.025)+color*smoothstep(.35,.75,vLight),1.);\n#include <colorspace_fragment>\n}'});
 let stopped=false,timer;const raw=document.createElement('canvas');raw.width=1280;raw.height=720;const rc=raw.getContext('2d');
 async function frame(){try{if(!document.hidden){const response=await fetch('/api/frame',{cache:'no-store'});if(!response.ok)throw Error('映像待ち');const bitmap=await createImageBitmap(await response.blob());rc.drawImage(bitmap,0,0,1280,720);bitmap.close();c.globalCompositeOperation='source-over';c.fillStyle='#000';c.fillRect(0,0,1280,720);c.drawImage(warp.paint(raw,state.source),0,0);c.globalCompositeOperation='multiply';c.drawImage(maskCanvas,0,0);c.globalCompositeOperation='source-over';texture.needsUpdate=true;if(mesh.material!==material){mesh.material.dispose();mesh.material=material;}status.textContent='オーキャンv2：全画像を顔へ変換中（光量・ぼけは未再現）';}}catch(error){status.textContent='ライブ映像を取得できません。'+error.message;}finally{if(!stopped)timer=setTimeout(frame,200);}}
 addEventListener('pagehide',()=>{stopped=true;clearTimeout(timer)},{once:true});frame();
}
