from pathlib import Path
import concurrent.futures, hashlib, json, time, urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAPERS = [
 ('P23_FutureOptions_2024','Generalization of Temporal Logic Tasks via Future Dependent Options','https://easychair.org/publications/preprint/FJpq','https://easychair.org/publications/preprint/FJpq/download'),
 ('P19_FORM_AAMAS2025','FORM: Learning Expressive and Transferable First-Order Logic Reward Machines','https://arxiv.org/abs/2501.00364','https://arxiv.org/pdf/2501.00364'),
 ('P20_RewardTranslation_ICML2025','Reward Translation via Reward Machine in Semi-Alignable MDPs','https://proceedings.mlr.press/v267/hua25a.html','https://raw.githubusercontent.com/mlresearch/v267/main/assets/hua25a/hua25a.pdf'),
 ('P21_SpatialOptions_2025','Learning Spatially Refined Sub-Policies for Temporal Task Composition in Continuous RL','https://openreview.net/pdf/c33d0d4c87e2287631c3d7012fb4eaf5cddf5eb5.pdf','https://bnaic2025.unamur.be/accepted-submissions/accepted_oral/007%20-%20Learning%20Spatially%20Refined%20Sub-Policies%20for%20%20Temporal%20Task%20Composition%20in%20Continuous%20RL.pdf'),
 ('P22_TemporalCausality_2025','Expediting Reinforcement Learning by Incorporating Knowledge About Temporal Causality in the Environment','https://arxiv.org/abs/2510.15456','https://arxiv.org/pdf/2510.15456'),
 ('P16_LOF_ICML2021','The Logical Options Framework','https://proceedings.mlr.press/v139/araki21a.html','https://proceedings.mlr.press/v139/araki21a/araki21a.pdf'),
 ('P17_PyCRM_SoftwareX2026','PyCRM: A Python library for reward machine-based reinforcement learning','https://www.raillab.org/publication/bester-2026-pycrm/','https://www.raillab.org/publication/bester-2026-pycrm/bester-2026-pycrm.pdf'),
 ('P18_LearningRM_POMDP','Learning Reward Machines: A Study in Partially Observable Reinforcement Learning','https://arxiv.org/abs/2112.09477','https://arxiv.org/pdf/2112.09477'),
 ('P01_ISA_JAIR2021','Induction and Exploitation of Subgoal Automata for Reinforcement Learning','https://arxiv.org/abs/2009.03855','https://arxiv.org/pdf/2009.03855'),
 ('P02_RewardMachines_JAIR2022','Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning','https://arxiv.org/abs/2010.03950','https://arxiv.org/pdf/2010.03950'),
 ('P03_JIRP_ICAPS2020','Joint Inference of Reward Machines and Policies for Reinforcement Learning','https://arxiv.org/abs/1909.05912','https://arxiv.org/pdf/1909.05912'),
 ('P04_HRM_ICML2023','Hierarchies of Reward Machines','https://proceedings.mlr.press/v202/furelos-blanco23a.html','https://proceedings.mlr.press/v202/furelos-blanco23a/furelos-blanco23a.pdf'),
 ('P05_TALearner_ECAI2023','Learning Task Automata for Reinforcement Learning using Hidden Markov Models','https://arxiv.org/abs/2208.11838','https://arxiv.org/pdf/2208.11838'),
 ('P06_CPREP','Contextual Pre-planning on Reward Machine Abstractions for Enhanced Transfer in Deep Reinforcement Learning','https://arxiv.org/abs/2307.05209','https://arxiv.org/pdf/2307.05209'),
 ('P07_HER_NeurIPS2017','Hindsight Experience Replay','https://arxiv.org/abs/1707.01495','https://arxiv.org/pdf/1707.01495'),
 ('P08_PolicySketches_ICML2017','Modular Multitask Reinforcement Learning with Policy Sketches','https://proceedings.mlr.press/v70/andreas17a.html','https://proceedings.mlr.press/v70/andreas17a/andreas17a.pdf'),
 ('P09_UVFA_ICML2015','Universal Value Function Approximators','https://proceedings.mlr.press/v37/schaul15.html','https://proceedings.mlr.press/v37/schaul15.pdf'),
 ('P10_NoisyRM_NeurIPS2024','Reward Machines for Deep RL in Noisy and Uncertain Environments','https://arxiv.org/abs/2406.00120','https://arxiv.org/pdf/2406.00120'),
 ('P11_PartialSemantics_AI2024','Joint Learning of Reward Machines and Policies in Environments with Partially Known Semantics','https://arxiv.org/abs/2204.11833','https://arxiv.org/pdf/2204.11833'),
 ('P12_AFRAI','Active Finite Reward Automaton Inference and Reinforcement Learning Using Queries and Counterexamples','https://arxiv.org/abs/2006.15714','https://arxiv.org/pdf/2006.15714'),
 ('P13_Shielding_AAAI2018','Safe Reinforcement Learning via Shielding','https://ojs.aaai.org/index.php/AAAI/article/view/11797','https://cdn.aaai.org/ojs/11797/11797-13-15325-1-2-20201228.pdf'),
 ('P14_RawTrajectoryRM_2026','Active Reward Machine Inference From Raw State Trajectories','https://arxiv.org/abs/2604.07480','https://arxiv.org/pdf/2604.07480'),
 ('P15_SymbolicRM_2026','Reinforcement Learning with Symbolic Reward Machines','https://arxiv.org/abs/2603.03068','https://arxiv.org/pdf/2603.03068'),
]

def download(p):
    ident,title,url,pdf_url=p
    path=ROOT/'papers'/f'{ident}.pdf'
    r=dict(id=ident,title=title,landing_url=url,pdf_url=pdf_url,file=str(path.relative_to(ROOT)))
    try:
        if not path.exists():
            req=urllib.request.Request(pdf_url,headers={'User-Agent':'Mozilla/5.0 (academic literature review)'})
            with urllib.request.urlopen(req,timeout=60) as resp:data=resp.read()
            if not data.startswith(b'%PDF'):raise ValueError('Response is not a PDF')
            path.write_bytes(data)
        data=path.read_bytes()
        r.update(status='downloaded',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:r.update(status='failed',error=str(exc))
    print(ident,r['status'],flush=True)
    return r

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(download,PAPERS))
    (ROOT/'paper_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
