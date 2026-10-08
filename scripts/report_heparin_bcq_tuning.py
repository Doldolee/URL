"""Read the completed Heparin BCQ study without changing reports or weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    folder = args.project_root/'outputs/heparin_mimic3_bcq_best_20261007'
    receipt = json.loads((folder/'completion_receipt.json').read_text())
    if receipt['status'] != 'complete':
        raise ValueError('The study has no completion receipt')
    for relative, digest in receipt['artifact_sha256'].items():
        if sha256_file(folder/relative) != digest:
            raise ValueError(f'Completed artifact changed: {relative}')
    selection = json.loads((folder/'selected/selection.json').read_text())
    evaluation = json.loads((folder/'selected/evaluation_seed42.json').read_text())
    print(json.dumps({'selected': selection['selected'],
                      'checkpoint_sha256': sha256_file(folder/'selected/policy_best.pth'),
                      'estimates': {name: evaluation['ope'][name] for name in ['dr', 'wdr', 'wis']},
                      'fqe': evaluation['fqe'],
                      'trajectory_ess': evaluation['ope']['weights']['trajectory_ess'],
                      'report': str(folder/'README.md')}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
