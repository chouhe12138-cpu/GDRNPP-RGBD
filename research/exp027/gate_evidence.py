"""Require one completed, matching EXP027 batch48/EGL gate before formal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def matching_gates(root, commit, config, scale, hierarchy):
    matches = []
    for metadata_path in sorted(Path(root).glob('RUN-*-gate-s42-*/run_metadata.json')):
        run = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('mode') != 'gate' or metadata.get('source_commit') != commit \
                    or metadata.get('config') != config:
                continue
            report = json.loads((run / 'gate' / 'report.json').read_text())
            exit_code = (run / 'exit_code').read_text().strip()
        except (OSError, ValueError):
            continue
        try:
            valid = (exit_code == '0' and report.get('status') == 'PASS'
                     and (report.get('config') == config or report.get('config', '').endswith('/' + config))
                     and float(report.get('amp_init_scale', -1)) == float(scale)
                     and report.get('hierarchy_sha256') == hierarchy
                     and int(report.get('batch_size', -1)) == 48
                     and int(report.get('steps', -1)) == 8
                     and report.get('amp_skipped_steps') == 0
                     and report.get('checkpoint_roundtrip') == 'PASS')
        except (TypeError, ValueError):
            valid = False
        if valid:
            matches.append(run)
    return matches


def require_unique_gate(root, commit, config, scale, hierarchy):
    matches = matching_gates(root, commit, config, scale, hierarchy)
    if len(matches) != 1:
        candidates = '\n'.join(f'CANDIDATE {run}' for run in matches)
        raise RuntimeError(f'Expected one matching EXP027 batch48 gate, found {len(matches)}'
                           + (f'\n{candidates}' if candidates else ''))
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('commit')
    parser.add_argument('config')
    parser.add_argument('scale', type=float)
    parser.add_argument('hierarchy')
    args = parser.parse_args()
    print('EXP027_GATE_EVIDENCE PASS', require_unique_gate(
        args.root, args.commit, args.config, args.scale, args.hierarchy))


if __name__ == '__main__':
    main()
