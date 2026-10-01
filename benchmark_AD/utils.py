"""Trace ingestion, reproducible room splits, prediction policy and run artifacts."""
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def predict_scores(args, scores, labels=None, probability=False):
    """No test labels used by default. Uncalibrated scores use an explicit top-k budget."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite anomaly scores')
    if getattr(args, 'threshold_policy', 'fixed') == 'oracle':
        if labels is None:
            raise ValueError('Oracle policy requires labels')
        k = int(np.asarray(labels).sum())
    elif probability:
        return (scores >= 0.5).astype(int)
    else:
        k = int(np.ceil(len(scores) * getattr(args, 'anomaly_fraction', 0.1)))
    result = np.zeros(len(scores), dtype=int)
    if k:
        result[np.argsort(-scores, kind='stable')[:k]] = 1
    return result


def stable_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def normalize_domain(value):
    return str(value).lower().replace('_', '-').replace('live-code-bench', 'livecodebench')


def load_traces_to_df_EscapeRoom_v3(folder_path, trace_type, trace_dataset_name):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f'Trace directory does not exist: {folder}')
    expected = 'gsm-hard' if trace_dataset_name == 'EscapeRoom_v3_gsm_hard' else 'livecodebench'
    rows = []
    for file in sorted(folder.rglob('*.json')):
        with file.open(encoding='utf-8') as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or not {'config', 'room', 'trace'} <= obj.keys():
            continue  # Ignore manifests/status files, never infer labels from them.
        try:
            config = obj['config']
            benchmark = config.get('benchmark', {})
            domain = benchmark.get('clue_domain')
            if domain is None:
                domain = next((part for part in file.parts if normalize_domain(part) in {'gsm-hard', 'livecodebench'}), None)
            if normalize_domain(domain) != expected:
                continue
            tooling = benchmark.get('unit_tools')
            if tooling is None:
                tooling = False if '_no_tool_' in file.parent.name else True if '_tool_' in file.parent.name else None
            if not isinstance(tooling, bool):
                raise ValueError('Missing or non-boolean config.benchmark.unit_tools')
            if tooling != (trace_type == 'tool'):
                continue
            escaped = obj.get('escaped')
            if not isinstance(escaped, bool):
                suffix = file.stem.rsplit('_', 1)[-1]
                if suffix not in {'escaped', 'failed'}:
                    raise ValueError('Missing escaped boolean and no filename label')
                escaped = suffix == 'escaped'
            if not isinstance(obj['trace'], list):
                raise ValueError('trace must be a list')
            source = file.relative_to(folder).as_posix()
            rows.append({
                'model_name': config['llm']['model'],
                'room_number': str(obj['room']['room_id']),
                'difficulty': str(obj['room']['difficulty']),
                'temperature': float(config['llm']['temperature']),
                'task': expected, 'trace_type': trace_type,
                'label': int(not escaped), 'id': stable_id(source),
                'source_file': source, 'trace': obj,
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid trace {file}: {exc}') from exc
    if not rows:
        raise ValueError(f'No traces matched domain={expected}, tooling={trace_type} under {folder}')
    return pd.DataFrame(rows)


def action_label(step, policy, source):
    status = (step.get('verification') or {}).get('status')
    if status == 'correct':
        return 0
    if status == 'wrong':
        return 1
    if status is None and step.get('schema_errors'):
        return 1  # Explicit schema failure, even when the verifier never ran.
    if policy == 'anomaly':
        return 1
    if policy == 'drop':
        return None
    raise ValueError(f'Unknown verification status {status!r} in {source}, action {step.get("action_id")}; choose --unknown-status anomaly or drop explicitly')


def convert_traces_action_steps(df, unknown_status='error'):
    rows = []
    empty = dropped = 0
    for _, row in df.iterrows():
        obj = row['trace']
        context = ''
        empty += not bool(obj['trace'])
        for step_num, step in enumerate(obj['trace']):
            message = f"{step.get('agent', '')}: {step.get('output_message', step.get('message', ''))}"
            context = (context + ' ' + message).strip()
            label = action_label(step, unknown_status, row['source_file'])
            if label is None:
                dropped += 1
                continue
            action_id = step.get('action_id', '')
            rows.append({
                'step': step, 'step_label': label, 'room_id': row['room_number'],
                'difficulty': row['difficulty'], 'puzzle_id': step.get('puzzle_id', ''),
                'action_id': action_id, 'model_name': row['model_name'],
                'temperature': row['temperature'], 'task': row['task'],
                'trace_type': row.get('trace_type', ''), 'source_file': row['source_file'],
                'step_num': step_num, 'trace_id': row['source_file'],
                'trace_label': row['label'], 'raw_trace': obj, 'input': step.get('input', ''),
                'step_trace': step, 'message_context': context,
                'row_id': stable_id(f"{row['source_file']}|{step_num}|{action_id}"),
            })
    if empty or dropped:
        warnings.warn(f'Action dataset omitted {empty} empty traces and {dropped} unknown-status actions')
    if not rows:
        raise ValueError('No evaluable actions after filtering')
    return pd.DataFrame(rows)


def get_train_test_indexes_action_level(args, df):
    label = 'step_label' if args.dataset_level == 'action' else 'label'
    room = 'room_id' if args.dataset_level == 'action' else 'room_number'
    # JSON tuple avoids ambiguous separator collisions. Preserve the full room ID.
    df['room_group'] = [json.dumps([str(t), str(r), str(d)]) for t, r, d in zip(df['task'], df[room], df['difficulty'])]
    groups = df.groupby('room_group', sort=True)[label].max()
    try:
        train_rooms, test_rooms = train_test_split(groups.index, test_size=args.test_size, stratify=groups.values, random_state=args.seed)
    except ValueError as exc:
        raise ValueError(f'Cannot stratify {len(groups)} room groups (class counts {groups.value_counts().to_dict()}); use more rooms or adjust --test_size. {exc}') from exc
    assert set(train_rooms).isdisjoint(test_rooms)
    return df['room_group'].isin(train_rooms), df['room_group'].isin(test_rooms)


def get_train_test_indexes_trace_level(args, labels):
    # Legacy utility; EscapeRoom setup uses room grouping at BOTH levels.
    return train_test_split(np.arange(len(labels)), test_size=args.test_size, stratify=labels, random_state=args.seed)


def setup_datasets(args):
    if args.trace_dataset_name.startswith('EscapeRoom_v3_'):
        df = load_traces_to_df_EscapeRoom_v3(args.path_traces, args.trace_type, args.trace_dataset_name)
    elif args.trace_dataset_name == 'EscapeRoom_v2':
        df = load_traces_to_df_EscapeRoom_v2(args.path_traces)
    elif args.trace_dataset_name == 'gsm_hard':
        if args.dataset_level != 'trace':
            raise ValueError('Legacy gsm_hard supports trace-level loading only')
        df = load_traces_to_df_gsm_hard(args.path_traces).rename(columns={'Temperature': 'temperature', 'MAS': 'model_name'})
    else:
        raise ValueError(f'Unknown dataset {args.trace_dataset_name}')
    if df.empty:
        raise ValueError(f'No traces in {args.path_traces}')
    if 'task' not in df:
        df['task'] = args.trace_dataset_name
    if 'source_file' not in df:
        raise ValueError('Loader did not retain source file identity')
    if 'room_number' not in df:
        df['room_number'] = df['id'].astype(str)
        df['difficulty'] = 'legacy'
    df = df[df['temperature'].isin([float(t) for t in args.temp_llm])].copy()
    if args.llm_in_mas:
        df = df[df['model_name'].isin(args.llm_in_mas)].copy()
    if df.empty:
        raise ValueError(f'No traces after filtering models={args.llm_in_mas}, temperatures={args.temp_llm}')
    if args.dataset_level == 'action':
        df = convert_traces_action_steps(df, getattr(args, 'unknown_status', 'error'))
    else:
        df['row_id'] = df['source_file'].map(stable_id)
    train_idx, test_idx = get_train_test_indexes_action_level(args, df)
    train_df, test_df = df[train_idx].reset_index(drop=True), df[test_idx].reset_index(drop=True)
    label = 'step_label' if args.dataset_level == 'action' else 'label'
    for split, frame in [('train', train_df), ('test', test_df)]:
        if frame[label].nunique() != 2:
            raise ValueError(f'{split} has only one label class; metrics/training require normal and anomalous samples. Adjust data or split.')
        print(f'{split}: {len(frame)} {args.dataset_level}s, {frame.room_group.nunique()} rooms, anomaly ratio={frame[label].mean():.4f}')
    return train_df, test_df


MANIFEST_COLUMNS = ['row_id', 'source_file', 'trace_id', 'room_id', 'room_number', 'room_group', 'difficulty', 'task', 'trace_type', 'model_name', 'temperature', 'puzzle_id', 'action_id', 'step_num', 'step_label', 'label']


def save_split_manifests(args, train_df, test_df):
    for split, df in [('train', train_df), ('test', test_df)]:
        manifest = df[[c for c in MANIFEST_COLUMNS if c in df]].copy()
        if not manifest.row_id.is_unique:
            raise ValueError(f'Duplicate sample IDs in {split}')
        manifest.to_csv(Path(args.run_dir) / f'{split}_manifest.csv', index=False)
    args._test_manifest = test_df[[c for c in MANIFEST_COLUMNS if c in test_df]].copy()


def log_results(args, epoch, epoch_total, f1, acc, auc, bal_acc, model=None, gt_labels=None, pred_labels=None, pred_scores=None, train_ids=None, test_ids=None):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([{'epoch': epoch, 'f1': f1, 'acc': acc, 'auc': auc, 'bal_acc': bal_acc,
                             'threshold_policy': getattr(args, 'threshold_policy', 'fixed'),
                             'anomaly_fraction': getattr(args, 'anomaly_fraction', 0.1)}])
    path = run_dir / 'metrics.csv'
    metrics.to_csv(path, mode='a', header=not path.exists(), index=False)
    if epoch != epoch_total - 1:
        return
    if test_ids is None:
        raise ValueError('Method must supply test row IDs to log_results')
    predictions = pd.DataFrame({'test_id': np.asarray(test_ids).reshape(-1),
                                'gt_label': np.asarray(gt_labels).reshape(-1),
                                'pred_label': np.asarray(pred_labels).reshape(-1),
                                'pred_score': np.asarray(pred_scores).reshape(-1)})
    if not predictions.test_id.is_unique:
        raise ValueError('Duplicate prediction IDs')
    manifest = getattr(args, '_test_manifest', None)
    if manifest is None:
        manifest = pd.read_csv(run_dir / 'test_manifest.csv', dtype={'row_id': str})
    predictions = predictions.merge(manifest, left_on='test_id', right_on='row_id', how='left', validate='one_to_one', indicator=True)
    if not predictions['_merge'].eq('both').all():
        raise ValueError('Predictions contain IDs absent from the test manifest')
    label_col = 'step_label' if 'step_label' in predictions else 'label'
    if not np.array_equal(predictions['gt_label'].to_numpy(), predictions[label_col].to_numpy()):
        raise ValueError('Prediction labels do not match manifest IDs (sample order mismatch)')
    predictions.drop(columns=['_merge']).to_csv(run_dir / 'predictions.csv', index=False)
    if model is not None:
        import torch
        torch.save(model.state_dict(), run_dir / 'model.pt')
    print(f'Predictions: {run_dir / "predictions.csv"}')
def load_traces_to_df_EscapeRoom_v2(folder_path):
    folder = Path(folder_path)
    rows = []

    id_temp = 0  # each trace should have its own id
    for file in sorted(folder.rglob("*.json")):

        if not file.stem[0].isdigit():
            continue

        filename = file.stem
        parts = filename.split("_")

        # Extract metadata
        room_num = parts[2]
        difficulty = parts[3]
        id = id_temp
        temperature = float(parts[5])
        label = parts[-1].lower()
        model_name = file.parent.name

        id_temp += 1

        # Convert label to numeric (0=inlier, 1=outlier)
        if label == "escaped":
            numeric_label = 0
        elif label == "failed":
            numeric_label = 1
        else: 
            raise ValueError(f"Label {label} not recognized for EscapeRoom_v2 dataset")
    
        
        # Load JSON trace
        with open(file, "r", encoding="utf-8") as f:
            trace_obj = json.load(f)

        rows.append({
            "model_name": model_name,
            "room_number": room_num,
            "difficulty": difficulty,
            "temperature": temperature,
            "label": numeric_label,
            "id": id,
            "source_file": file.relative_to(folder).as_posix(),
            "trace": trace_obj
        })

    return pd.DataFrame(rows)


def load_traces_to_df_gsm_hard(folder_path):
    folder = Path(folder_path)
    rows = []

    for file in sorted(folder.glob("*.json")):

        if not file.stem[0].isdigit():
            continue

        filename = file.stem
        parts = filename.split("_")

        # Extract metadata
        task_dataset = parts[1]
        mas = parts[2]
        label = parts[-1].lower()
        id = parts[5]
        temp = float(parts[7])

        # Convert label to numeric (your convention: 0=inlier, 1=outlier)
        if label == "abnormal" or label == "parseerror":
            if parts[-2].lower() == "normal":  # some traces that have parseerror might still be normal
                numeric_label = 0
            else:
                numeric_label = 1
        elif label == "normal":
            numeric_label = 0
        else:
            numeric_label = None
        
        # Load JSON trace
        with open(file, "r", encoding="utf-8") as f:
            trace_obj = json.load(f)

        rows.append({
            "MAS": mas,
            "Task Dataset": task_dataset,
            "Temperature": temp,
            "label": numeric_label,
            "id": id,
            "source_file": file.relative_to(folder).as_posix(),
            "trace": trace_obj
        })

    return pd.DataFrame(rows)




def sequence_row_ids(df, limit=100):
    return [row_id for _, group in df.groupby('trace_id', sort=False)
            for row_id in group.sort_values('step_num').head(limit).row_id]
