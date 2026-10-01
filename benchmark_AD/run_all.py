"""Sequential, failure-isolated AD experiment matrix with per-job logs."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_AD.registry import METHODS, DATASETS, default_level, validate_level

REPO_ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', choices=['all', *METHODS], default=['all'])
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS[:2]))
    p.add_argument('--trace-types', nargs='+', choices=['tool', 'no_tool'], default=['tool', 'no_tool'])
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--dataset-level', choices=['auto', 'action', 'trace'], default='auto')
    p.add_argument('--path-traces', type=Path, default=REPO_ROOT / 'output/traces')
    p.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output/ad')
    p.add_argument('--temperatures', nargs='+', type=float, default=[0.0])
    p.add_argument('--models', nargs='+')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--unknown-status', choices=['error', 'anomaly', 'drop'], default='error')
    p.add_argument('--threshold-policy', choices=['fixed', 'oracle'], default='fixed')
    p.add_argument('--anomaly-fraction', type=float, default=0.1)
    p.add_argument('--occ-training', choices=['clean', 'polluted'], default='clean')
    p.add_argument('--perc-outliers-train', type=float, default=0.2)
    p.add_argument('--epochs', type=int, help='Optional neural epoch override')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Write planned commands without running jobs')
    p.add_argument('--skip-llm-inference', action='store_true',
                   help='Export Gemma inputs only; by default Gemma runs in this same Python environment')
    p.add_argument('--llm-models', nargs='+', default=['gemma-4-E2B'], choices=['gemma-4-E2B', 'gemma-4-E4B', 'gemma-4-31B', 'gemma-4-26B-A4B'])
    return p


def run_logged(command, log_path):
    with log_path.open('w') as log:
        try:
            return subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        except OSError as exc:
            log.write(f'Could not start interpreter/process: {exc}\n')
            return 127


def main(argv=None):
    args = parser().parse_args(argv)
    methods = list(METHODS) if 'all' in args.methods else list(dict.fromkeys(args.methods))
    batch = args.output_dir.expanduser().resolve() / ('batch_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8])
    batch.mkdir(parents=True)
    jobs = []
    for dataset, tooling, seed, method in itertools.product(args.datasets, args.trace_types, args.seeds, methods):
        level = default_level(method) if args.dataset_level == 'auto' else args.dataset_level
        validate_level(method, level)
        run_dir = batch / dataset / tooling / level / method / f'seed_{seed}'
        cmd = [sys.executable, '-u', '-m', 'benchmark_AD.main', '--method', method,
               '--trace_dataset_name', dataset, '--trace_type', tooling, '--seed', str(seed),
               '--dataset_level', level, '--path_traces', str(args.path_traces.expanduser().resolve()),
               '--run-dir', str(run_dir), '--test_size', str(args.test_size),
               '--unknown-status', args.unknown_status, '--threshold-policy', args.threshold_policy,
               '--anomaly-fraction', str(args.anomaly_fraction), '--occ_training', args.occ_training,
               '--perc_outliers_train', str(args.perc_outliers_train), '--temp_llm', *map(str, args.temperatures)]
        if args.epochs is not None:
            cmd += ['--epochs', str(args.epochs)]
        if args.models:
            cmd += ['--llm_in_mas', *args.models]
        if args.prepare_only:
            cmd += ['--prepare-only']
        jobs.append({'method': method, 'dataset': dataset, 'trace_type': tooling, 'seed': seed,
                     'level': level, 'run_dir': str(run_dir), 'command': cmd, 'status': 'planned'})
    manifest = batch / 'batch_manifest.json'
    def save():
        manifest.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
        rows = []
        for job in jobs:
            targets = job.get('inference') or [job]
            for target in targets:
                directory = Path(target['run_dir'])
                row = {k: job[k] for k in ('method', 'dataset', 'trace_type', 'seed', 'level', 'status')}
                row['judge_model'] = target.get('model', '')
                if 'model' in target:
                    row['status'] = 'completed' if target['returncode'] == 0 else 'failed'
                row['run_dir'] = str(directory)
                row['predictions'] = str(directory / 'predictions.csv') if (directory / 'predictions.csv').exists() else ''
                if (directory / 'metrics.csv').exists():
                    with (directory / 'metrics.csv').open() as f:
                        metrics = list(csv.DictReader(f))
                    if metrics:
                        row.update({k: metrics[-1].get(k, '') for k in ('f1', 'acc', 'auc', 'bal_acc')})
                rows.append(row)
        with (batch / 'batch_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'judge_model', 'dataset', 'trace_type', 'seed', 'level', 'status', 'f1', 'acc', 'auc', 'bal_acc', 'run_dir', 'predictions'])
            writer.writeheader()
            writer.writerows(rows)
    save()
    print(f'Batch: {batch}\nJobs: {len(jobs)}', flush=True)
    if args.dry_run:
        return 0
    for index, job in enumerate(jobs, 1):
        run_dir = Path(job['run_dir'])
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        # Keep logs beside the run until main creates its exclusive output directory.
        log = run_dir.parent / f'{run_dir.name}.log'
        job['status'] = 'running'
        save()
        print(f'[{index}/{len(jobs)}] {job["method"]} {job["dataset"]} {job["trace_type"]} seed={job["seed"]}', flush=True)
        start = time.monotonic()
        code = run_logged(job['command'], log)
        job['returncode'] = code
        job['elapsed_seconds'] = round(time.monotonic() - start, 2)
        job['status'] = 'failed' if code else 'prepared' if args.prepare_only else 'exported' if job['method'] == 'gemma4' else 'completed'
        if run_dir.is_dir():
            log.rename(run_dir / 'run.log')
            job['log'] = str(run_dir / 'run.log')
        else:
            job['log'] = str(log)
        if code == 0 and job['method'] == 'gemma4' and not args.skip_llm_inference and not args.prepare_only:
            job['inference'] = []
            for model in args.llm_models:
                infer_dir = run_dir / model
                command = [sys.executable, '-u', '-m', 'benchmark_AD.methods.llm_as_judge.run_gemma_4',
                           '--data-dir', str(run_dir / 'data'), '--run-dir', str(infer_dir), '--method', model]
                llm_code = run_logged(command, run_dir / f'{model}.log')
                job['inference'].append({'model': model, 'returncode': llm_code, 'run_dir': str(infer_dir)})
            job['status'] = 'failed' if any(x['returncode'] for x in job['inference']) else 'completed'
        save()
        print(f'  {job["status"]}; log: {job["log"]}', flush=True)
    failures = sum(job['status'] == 'failed' for job in jobs)
    print(f'Finished: {len(jobs) - failures} successful/prepared/exported, {failures} failed. {manifest}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
