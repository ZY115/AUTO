import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const ROOT='/Users/yuhang/Downloads/why TL';
const WORK=ROOT+'/research/2026-09-15-project-slides';
const SKILL='/Users/yuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const FONT='Arial Unicode MS';
const C={ink:'#202B33',muted:'#687780',line:'#D9E0E4',blue:'#2176AD',light:'#EAF3F9',orange:'#C47A35',gray:'#94A1AA',red:'#AE514B',green:'#39806D',white:'#FFFFFF'};
const FINAL=WORK+'/output/任务结构研究_口播配套_可编辑.pptx';
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:WORK+'/.build/candidate.pptx',finalPath:FINAL,
 pythonExecutable:'/Users/yuhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:14,requiredNativeChartOwnerSlides:[2,7,9,10,11,12,13],requiredNativeTableOwnerSlides:[],
 materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'portable',fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 receiptPath:WORK+'/.build/validation_v2.json'});
console.log(JSON.stringify(result,null,2));
