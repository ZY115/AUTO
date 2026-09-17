"""Small native runner with an append-only run record and explicit arm name."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from native_hrm.y10 import IHSAAlgorithmHRLPerState
from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--records', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    arm = config.get('algorithm')
    if arm not in ('ihsa-hrl', 'ihsa-hrl-perstate'):
        raise ValueError('This runner handles native tabular Y11/Y10 only.')
    if config.get('state_format') != 'tabular':
        raise ValueError('Use the author runner for neural arms; this runner is tabular.')
    if config.get('checkpoint_enable', False):
        raise ValueError('Checkpoint resume is not wired into this runner; do not silently restart.')
    folder = Path(config['folder_name']).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing run: {folder}')
    config['folder_name'] = str(folder)
    record = dict(config=config, command=[sys.executable]+sys.argv, status='running',
                  python=sys.version, started=time.time(),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((BASE/'native_hrm').glob('*.py'))},
                  upstream_commit=subprocess.check_output(
                      ['git','-C',str(ROOT/'external/hrm-learning'),'rev-parse','HEAD'],text=True).strip())
    records = Path(args.records)
    records.parent.mkdir(parents=True, exist_ok=True)
    def save():
        with records.open('a') as f:
            f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    save()
    try:
        cls = IHSAAlgorithmHRLPerState if arm == 'ihsa-hrl-perstate' else IHSAAlgorithmHRLTabular
        algo = cls(config)
        algo.run(False)
        record['status'] = 'complete'
    except BaseException as e:
        record.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        record['finished'] = time.time();save()


if __name__ == '__main__':
    main()
