"""Run one AD method; python -m benchmark_AD.main --help."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import random
import sys
import traceback
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_AD.registry import METHODS, DATASETS, default_level, validate_level

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', default=5, type=int)
    p.add_argument('--method', default='random_forest', choices=METHODS)
    p.add_argument('--trace_dataset_name', default=DATASETS[0], choices=DATASETS)
    p.add_argument('--trace_type', default='tool', choices=['tool', 'no_tool'])
    p.add_argument('--path_traces', type=Path, default=None, help='Trace root or one run directory; defaults to project output/traces')
    p.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output/ad')
    p.add_argument('--run-dir', type=Path, help='Exact NEW run directory (normally assigned by run_all)')
    p.add_argument('--test_size', default=0.2, type=float)
    p.add_argument('--dataset_level', choices=['action', 'trace'], help='Default: action, or trace for graph classifiers')
    p.add_argument('--llm_in_mas', nargs='+', default=None, help='Model allowlist; default includes all models')
    p.add_argument('--temp_llm', nargs='+', default=[0.0], type=float)
    p.add_argument('--occ_training', default='clean', choices=['clean', 'polluted'])
    p.add_argument('--perc_outliers_train', default=0.2, type=float)
    p.add_argument('--unknown-status', default='error', choices=['error', 'anomaly', 'drop'], help='Action status other than correct/wrong')
    p.add_argument('--threshold-policy', default='fixed', choices=['fixed', 'oracle'], help='fixed: probability >=0.5 or explicit top fraction; oracle: legacy test-label threshold')
    p.add_argument('--anomaly-fraction', default=0.1, type=float, help='Top fraction for uncalibrated scores under fixed policy (transductive ranking)')
    p.add_argument('--epochs', type=int, help='Override neural training epochs (including DeepSAD pretraining)')
    p.add_argument('--prepare-only', action='store_true', help='Validate inputs and save split manifests without loading ML models')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.dataset_level = args.dataset_level or default_level(args.method)
    validate_level(args.method, args.dataset_level)
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('--epochs must be positive')
    if not 0 < args.test_size < 1:
        raise ValueError('--test_size must be between 0 and 1')
    if not 0 <= args.anomaly_fraction <= 1 or not 0 <= args.perc_outliers_train <= 1:
        raise ValueError('Fractions must be in [0, 1]')
    args.path_traces = (args.path_traces or REPO_ROOT / 'output/traces').expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
    args.run_dir = (args.run_dir or args.output_dir / run_id / args.trace_dataset_name / args.trace_type / args.dataset_level / args.method / f'seed_{args.seed}').expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    print(f'Run directory: {args.run_dir}', flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    status = {'status': 'running', 'method': args.method}
    try:
        import numpy as np
        from benchmark_AD.utils import setup_datasets, save_split_manifests
        random.seed(args.seed)
        np.random.seed(args.seed)
        train_df, test_df = setup_datasets(args)
        save_split_manifests(args, train_df, test_df)
        if args.prepare_only:
            status['status'] = 'prepared'
        else:
            if args.method not in {'random', 'gemma4'}:
                import torch
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            module, name = METHODS[args.method]
            run = getattr(importlib.import_module('benchmark_AD.methods.' + module), name)
            if args.method == 'gemma4':
                run(args, 'zero_shot', train_df, test_df)
                status['status'] = 'exported'
            else:
                run(args, train_df, test_df)
                if not (args.run_dir / 'predictions.csv').exists():
                    raise RuntimeError('Method returned without final predictions.csv')
                status['status'] = 'completed'
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (args.run_dir / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        (args.run_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
