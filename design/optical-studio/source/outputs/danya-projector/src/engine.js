import {Renderer} from './renderer.js';
import {defaults,clone,validateState} from './math.js';
export class Engine{
 constructor(canvas,{onState=()=>{},onError=()=>{},onStatus=()=>{}}={}){
  this.renderer=new Renderer(canvas);this.state=defaults();this.view='projection';this.raw=false;this.onState=onState;this.onError=onError;this.onStatus=onStatus;this.dirty=true;this.sourceDirty=true;this.sourceId=0;this.seq=0;this.busy=false;
  this.normal=document.createElement('canvas');this.normal.width=720;this.normal.height=1000;this.normal.getContext('2d').fillRect(0,0,720,1000);this.scratch=document.createElement('canvas');this.worker=new Worker(new URL('./warp-worker.js',import.meta.url),{type:'module'});
  this.worker.onmessage=({data:r})=>{this.busy=false;if(r.id===this.seq){if(r.error)this.onError(r.error);else{this.renderer.setField(r);this.state=clone(this.pending);this.dirty=this.sourceDirty=true;this.lastWarpMS=r.ms;this.onState(clone(this.state),r);}}if(r.id!==this.seq)this.dispatch();};
  this.worker.onerror=e=>this.onError('変形計算に失敗しました：'+e.message);this.setState(this.state);this.renderer.upload(this.normal);this.frames=0;this.fps=0;this.lastStats=performance.now();this.loop();
 }
 setState(s){try{this.pending=validateState(s);this.seq++;if(s.calibration.mode==='v5')this.renderer.loadOptical().then(()=>{this.dirty=true;}).catch(e=>this.onError(e.message));if(!this.busy)this.dispatch();}catch(e){this.onError(e.message);}}
 dispatch(){this.busy=true;this.worker.postMessage({id:this.seq,state:this.pending});}
 async setSource(desc){
  const id=++this.sourceId;this.cleanupSource();this.sourceDesc=desc;this.lastTime=-1;this.sourceDirty=true;
  try{
   if(desc.kind==='feed'){this.media=new Image();this.feedTick(id);this.onStatus('オーキャン配信を接続中…');return;}
   if(desc.kind==='stream'||desc.kind==='video'){
    const v=document.createElement('video');v.muted=true;v.playsInline=true;v.loop=desc.kind==='video';v.preload='auto';this.media=v;
    if(desc.kind==='stream'){this.ownedStream=desc.stream.clone();v.srcObject=this.ownedStream;for(const t of desc.stream.getVideoTracks())t.addEventListener('ended',()=>{if(this.sourceId===id){this.onStatus('画面共有が終了しました。出力は最後のフレームです。');}},{once:true});}
    else {this.ownedURL=URL.createObjectURL(desc.file);v.src=this.ownedURL;}
    await v.play();if(id!==this.sourceId)return;this.onStatus(desc.name||'動画を再生中');
   }else{const img=new Image();if(desc.file){this.ownedURL=URL.createObjectURL(desc.file);img.src=this.ownedURL;}else img.src=desc.url||'assets/sample.jpg';await img.decode();if(id!==this.sourceId)return;this.media=img;this.onStatus(desc.name||'オーキャン参考画像（静止画）');}
   this.sourceDirty=true;
  }catch(e){if(id===this.sourceId)this.onError('映像を開けません：'+e.message);}
 }
 cleanupSource(){clearTimeout(this.feedTimer);if(this.media instanceof HTMLVideoElement){this.media.pause();this.media.srcObject=null;this.media.removeAttribute('src');this.media.load();}this.ownedStream?.getTracks().forEach(t=>t.stop());this.ownedStream=null;if(this.ownedURL)URL.revokeObjectURL(this.ownedURL);this.ownedURL=null;this.media=null;}
 async feedTick(id){if(id!==this.sourceId)return;try{const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());if(!s.live)throw Error('配信が停止しています。元アプリを起動するか、動画を読み込んでください。');const r=await fetch('/api/frame?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('フレームを取得できません');const b=await createImageBitmap(await r.blob());if(id!==this.sourceId){b.close();return;}this.media?.close?.();this.media=b;this.sourceDirty=true;this.onStatus('オーキャン配信 · 最大15 fps（元アプリが必要）');}catch(e){if(id===this.sourceId)this.onStatus(e.message);}if(id===this.sourceId)this.feedTimer=setTimeout(()=>this.feedTick(id),70);}
 normalize(){const m=this.media;if(!m)return false;const w=m.videoWidth||m.naturalWidth||m.width,h=m.videoHeight||m.naturalHeight||m.height;if(!w||!h)return false;
  const {rotation,flipX,flipY,crop}=this.state.source,side=rotation%180!==0,sw=side?h:w,sh=side?w:h;const c=this.scratch;if(c.width!==sw||c.height!==sh){c.width=sw;c.height=sh;}const x=c.getContext('2d');x.setTransform(1,0,0,1,0,0);x.clearRect(0,0,sw,sh);x.translate(sw/2,sh/2);x.scale(flipX?-1:1,flipY?-1:1);x.rotate(rotation*Math.PI/180);x.drawImage(m,-w/2,-h/2);x.setTransform(1,0,0,1,0,0);
  const z=this.normal.getContext('2d');z.imageSmoothingEnabled=true;z.imageSmoothingQuality='high';z.drawImage(c,crop[0]*sw,crop[1]*sh,crop[2]*sw,crop[3]*sh,0,0,720,1000);this.renderer.upload(this.normal);return true;
 }
 loop(){if(this.closed)return;try{const v=this.media instanceof HTMLVideoElement,changed=v&&this.media.readyState>=2&&this.media.currentTime!==this.lastTime;if(changed){this.lastTime=this.media.currentTime;this.sourceDirty=true;}if(this.sourceDirty&&this.normalize()){this.sourceDirty=false;this.dirty=true;}if(this.dirty){this.renderer.draw(this.state,this.view,this.raw);this.dirty=false;this.frames++;}const now=performance.now();if(now-this.lastStats>=1000){this.fps=this.frames*1000/(now-this.lastStats);this.frames=0;this.lastStats=now;}}catch(e){this.onError(e.message);}this.raf=requestAnimationFrame(()=>this.loop());}
 setView(view,raw=false){this.view=view;this.raw=raw;this.dirty=true;}
 destroy(){this.closed=true;cancelAnimationFrame(this.raf);this.worker.terminate();this.cleanupSource();}
}
