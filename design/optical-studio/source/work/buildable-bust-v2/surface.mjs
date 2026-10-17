export const profiles=[[-182,88,111,2],[-145,96,126,2],[-100,103,149,2],[-62,99,165,2],[-35,89,162,2],[-12,70,133,2],[12,52,82,2],[38,48,53,2]];
function profile(z){let i=0;while(i<profiles.length-2&&z>profiles[i+1][0])i++;const h=profiles[i+1][0]-profiles[i][0],t=(z-profiles[i][0])/h;return[1,2,3].map(k=>{const sec=j=>(profiles[j+1][k]-profiles[j][k])/(profiles[j+1][0]-profiles[j][0]);const slope=j=>{if(j===0)return sec(0);if(j===profiles.length-1)return sec(j-1);const a=sec(j-1),b=sec(j);return a*b<=0?0:2*a*b/(a+b);};return(2*t**3-3*t*t+1)*profiles[i][k]+(t**3-2*t*t+t)*h*slope(i)+(-2*t**3+3*t*t)*profiles[i+1][k]+(t**3-t*t)*h*slope(i+1);});}

export function point(z,t,inner){let[a,b,n]=profile(z);const c=Math.cos(t),s=Math.sin(t),y=b*Math.sign(s)*Math.abs(s)**(2/n),front=Math.max(0,c),back=Math.max(0,-c);
// Paired pectorals, central sternum valley, clavicular ridge and upper-back volume.
const sq=x=>x*x;
const pec=13*Math.exp(-sq((Math.abs(y)-66)/43))*Math.exp(-sq((z+83)/42));
const sternum=4*Math.exp(-sq(y/18))*Math.exp(-sq((z+75)/55));
const clavicle=5*Math.exp(-sq((z+22+Math.abs(y)*.18)/9))*Math.exp(-sq((Math.abs(y)-57)/50));
const scapula=4*Math.exp(-sq((Math.abs(y)-70)/62))*Math.exp(-sq((z+55)/55));
let x=15.7629+a*Math.sign(c)*Math.abs(c)**(2/n)+front**2*(pec-sternum+clavicle)-back**2*scapula;
let yy=y;if(inner){const e=.001,pa=point(z,t+e,false),pb=point(z,t-e,false),pc=point(z+e,t,false),pd=point(z-e,t,false),a=pa.map((v,i)=>v-pb[i]),b=pc.map((v,i)=>v-pd[i]),nn=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],len=Math.hypot(...nn),cosine=(nn[0]*c+nn[1]*s)/len,offset=3/Math.max(.15,cosine);x-=offset*c;yy-=offset*s;}return[x,yy,z];}

