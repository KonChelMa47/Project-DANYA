import * as THREE from 'three';

// Rasterize the complete front-facing source image into the physical projector UVs.
// Geometry changes when the optical layout changes; video frames only update a texture.
export function buildFaceWarpGeometry(face, projectionUV, illuminated) {
 const source=face.attributes.position,indices=face.userData.outerIndices||face.index.array;
 const bounds=new THREE.Box3();for(const i of indices)bounds.expandByPoint(new THREE.Vector3().fromBufferAttribute(source,i));
 const width=bounds.max.y-bounds.min.y,height=bounds.max.z-bounds.min.z;
 if(!(width>0&&height>0))throw Error('Face has no frontal extent');
 const positions=new Float32Array(source.count*3),uv=new Float32Array(source.count*2),triangles=[];
 for(let i=0;i<source.count;i++){
  positions[3*i]=2*projectionUV[2*i]-1;positions[3*i+1]=2*projectionUV[2*i+1]-1;
  uv[2*i]=(source.getY(i)-bounds.min.y)/width;uv[2*i+1]=(source.getZ(i)-bounds.min.z)/height;
 }
 for(let j=0;j<indices.length;j+=3){const a=indices[j],b=indices[j+1],c=indices[j+2];if(![a,b,c].every(i=>illuminated[i]>.5))continue;
  if(!face.userData.outerIndices){const ab=new THREE.Vector3().fromBufferAttribute(source,b).sub(new THREE.Vector3().fromBufferAttribute(source,a)),ac=new THREE.Vector3().fromBufferAttribute(source,c).sub(new THREE.Vector3().fromBufferAttribute(source,a));if(ab.cross(ac).x<=0)continue;}
  triangles.push(a,b,c);
 }
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(positions,3));g.setAttribute('uv',new THREE.BufferAttribute(uv,2));g.setIndex(triangles);g.userData.bounds=bounds;return g;
}

export class FaceWarp {
 constructor(){
  this.canvas=document.createElement('canvas');this.context=this.canvas.getContext('2d');
  this.texture=new THREE.CanvasTexture(this.canvas);this.texture.colorSpace=THREE.SRGBColorSpace;this.texture.minFilter=THREE.LinearFilter;this.texture.generateMipmaps=false;
  this.renderer=new THREE.WebGLRenderer({alpha:false,antialias:false,preserveDrawingBuffer:true});this.renderer.setSize(1280,720,false);this.renderer.setClearColor(0x000000,1);this.renderer.outputColorSpace=THREE.SRGBColorSpace;
  this.scene=new THREE.Scene();this.camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,10);this.camera.position.z=1;
  this.mesh=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshBasicMaterial({map:this.texture,side:THREE.DoubleSide,depthTest:false,depthWrite:false,toneMapped:false}));this.mesh.frustumCulled=false;this.scene.add(this.mesh);this.ready=false;
 }
 update(face,uv,lit){this.mesh.geometry.dispose();this.mesh.geometry=buildFaceWarpGeometry(face,uv,lit);this.ready=this.mesh.geometry.index.count>0;}
 invalidate(){this.ready=false;}
 paint(raw,settings){
  const sideways=settings.rotation%180!==0,w=sideways?raw.height:raw.width,h=sideways?raw.width:raw.height;
  if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}
  const c=this.context;c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,w,h);c.translate(w/2,h/2);c.scale(settings.flipX?-1:1,settings.flipY?-1:1);c.rotate(settings.rotation*Math.PI/180);c.drawImage(raw,-raw.width/2,-raw.height/2);c.setTransform(1,0,0,1,0,0);
  this.texture.needsUpdate=true;this.renderer.clear();if(this.ready)this.renderer.render(this.scene,this.camera);return this.renderer.domElement;
 }
}
