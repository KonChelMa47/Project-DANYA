import {Engine} from './engine.js';
const error=m=>document.getElementById('error').textContent=m;
const engine=new Engine(document.getElementById('output'),{onError:error});
window.danyaOutput=engine;
try{const {state,source}=window.opener.danyaBridge.snapshot();engine.setState(state);engine.setSource(source);}catch{error('調整画面の「投影ウィンドウを開く」から開いてください。');}
document.getElementById('full').onclick=()=>document.documentElement.requestFullscreen().catch(e=>error(e.message));
document.getElementById('black').onclick=()=>{const s=structuredClone(engine.state);s.blackout=!s.blackout;engine.setState(s);};
document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='f')document.getElementById('full').click();if(e.key.toLowerCase()==='b')document.getElementById('black').click();});
