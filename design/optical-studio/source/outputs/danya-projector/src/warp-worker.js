import {warpField} from './math.js';
self.onmessage=({data:{id,state}})=>{const start=performance.now();try{const r=warpField(state);self.postMessage({id,...r,ms:performance.now()-start},[r.data.buffer]);}catch(e){self.postMessage({id,error:e.message});}};
