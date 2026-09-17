import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='/Users/yuhang/Downloads/why TL/research/2026-09-15-project-slides';
const file=process.argv[2]||root+'/output/任务结构研究_口播配套.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(file));
await fs.mkdir(root+'/.build/final-previews',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(root+'/.build/final-previews/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
}console.log('Rendered final',p.slides.items.length);
