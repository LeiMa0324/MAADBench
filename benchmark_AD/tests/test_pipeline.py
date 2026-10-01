import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmark_AD.main import main
from benchmark_AD.run_all import main as run_all
from benchmark_AD.registry import METHODS, default_level
from benchmark_AD.utils import (setup_datasets, load_traces_to_df_EscapeRoom_v3,
    predict_scores, log_results, save_split_manifests, sequence_row_ids, action_label)


@pytest.fixture
def traces(tmp_path):
    root = tmp_path / 'traces'
    for model in ['model-a', 'model-b']:
        for room in range(12):
            # Six all-normal rooms, six with both normal and anomalous steps.
            failed = room >= 6
            data = {
                'config': {'llm': {'model': model, 'temperature': 0.0},
                           'benchmark': {'clue_domain': 'gsm-hard', 'unit_tools': True},
                           'agents': [{'role': 'solver', 'prompt': 'solve'}]},
                'room': {'room_id': f'room_{room:04d}', 'difficulty': 'easy'},
                'escaped': not failed,
                'trace': [{'agent': 'solver', 'action_id': f'room_{room}__a{i}', 'puzzle_id': 'p1',
                           'input': 'question', 'output_message': f'answer {i}',
                           'verification': {'status': 'wrong' if failed and i else 'correct'}} for i in range(2)]}
            # Both current flat and archived nested layouts; filenames need not encode a label.
            folder = root / (f'run_{model}' if model == 'model-a' else 'model-b/gsm_hard/run_old_tool_model-b')
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f'trace_{room}.json').write_text(json.dumps(data))
    (root / 'status.json').write_text('{"status":"done"}')
    return root


def args(root, **kwargs):
    values = dict(path_traces=root, trace_dataset_name='EscapeRoom_v3_gsm_hard', trace_type='tool',
                  temp_llm=[0.0], llm_in_mas=None, dataset_level='action', seed=3,
                  test_size=0.33, unknown_status='error')
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('level', ['action', 'trace'])
def test_room_split_and_stable_ids(traces, level):
    train, test = setup_datasets(args(traces, dataset_level=level))
    assert set(train.room_group).isdisjoint(test.room_group)
    assert train.row_id.is_unique and test.row_id.is_unique
    tr2, te2 = setup_datasets(args(traces, dataset_level=level))
    assert train.row_id.tolist() == tr2.row_id.tolist()
    assert test.row_id.tolist() == te2.row_id.tolist()
    all_rows = pd.concat([train, test]).set_index(['source_file'] + (['step_num'] if level == 'action' else []))
    tr3, te3 = setup_datasets(args(traces, dataset_level=level, llm_in_mas=['model-a']))
    subset = pd.concat([tr3, te3]).set_index(['source_file'] + (['step_num'] if level == 'action' else []))
    assert all_rows.loc[subset.index].row_id.tolist() == subset.row_id.tolist()


def test_metadata_filter_and_no_match(traces):
    train, test = setup_datasets(args(traces, llm_in_mas=['model-a']))
    assert set(pd.concat([train, test]).model_name) == {'model-a'}
    with pytest.raises(ValueError, match='No traces matched'):
        load_traces_to_df_EscapeRoom_v3(traces, 'no_tool', 'EscapeRoom_v3_gsm_hard')
    with pytest.raises(ValueError, match='after filtering'):
        setup_datasets(args(traces, temp_llm=[0.7]))


def test_unknown_status_policy():
    with pytest.raises(ValueError, match='Unknown verification'):
        action_label({}, 'error', 'x.json')
    assert action_label({}, 'drop', 'x.json') is None
    assert action_label({}, 'anomaly', 'x.json') == 1
    assert action_label({'schema_errors': {'answer': 'missing'}}, 'error', 'x.json') == 1


def test_fixed_predictions_independent_of_test_labels():
    config = SimpleNamespace(threshold_policy='fixed', anomaly_fraction=0.25)
    scores = [0.1, 0.2, 0.8, 0.9]
    assert np.array_equal(predict_scores(config, scores, [0]*4), predict_scores(config, scores, [1]*4))
    assert predict_scores(config, scores).sum() == 1
    assert predict_scores(config, scores, probability=True).sum() == 2
    config.threshold_policy = 'oracle'
    assert predict_scores(config, scores, [0]*4).sum() == 0
    assert predict_scores(config, scores, [1]*4).sum() == 4


def test_random_end_to_end(traces, tmp_path):
    output = tmp_path / 'run'
    assert main(['--method', 'random', '--path_traces', str(traces), '--run-dir', str(output), '--test_size', '0.33']) == 0
    pred = pd.read_csv(output / 'predictions.csv')
    manifest = pd.read_csv(output / 'test_manifest.csv')
    assert len(pred) == len(manifest)
    assert set(pred.test_id) == set(manifest.row_id)
    assert {'source_file', 'action_id', 'model_name', 'temperature', 'pred_score'} <= set(pred)
    assert json.loads((output / 'status.json').read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        main(['--method', 'random', '--run-dir', str(output)])


def test_prediction_join_rejects_wrong_order(traces, tmp_path):
    config = args(traces, run_dir=tmp_path)
    train, test = setup_datasets(config)
    save_split_manifests(config, train, test)
    with pytest.raises(ValueError, match='do not match'):
        log_results(config, 0, 1, 0, 0, 0, 0, gt_labels=1-test.step_label.to_numpy(),
                    pred_labels=test.step_label.to_numpy(), pred_scores=np.zeros(len(test)), test_ids=test.row_id.to_numpy())


def test_batch_continues_after_failure(traces, tmp_path, monkeypatch):
    import benchmark_AD.run_all as batch_module
    calls = []
    def fake_run(command, log):
        calls.append(command)
        log.write_text('test child log')
        return 1 if len(calls) == 1 else 0
    monkeypatch.setattr(batch_module, 'run_logged', fake_run)
    code = run_all(['--methods', 'random', 'random_forest', '--datasets', 'EscapeRoom_v3_gsm_hard',
                   '--trace-types', 'tool', '--path-traces', str(traces), '--output-dir', str(tmp_path / 'out')])
    assert code == 1 and len(calls) == 2
    manifest = json.loads(next((tmp_path / 'out').rglob('batch_manifest.json')).read_text())
    assert [x['status'] for x in manifest['jobs']] == ['failed', 'completed']


def test_all_methods_in_dry_run(tmp_path):
    run_all(['--dry-run', '--output-dir', str(tmp_path)])
    jobs = json.loads(next(tmp_path.rglob('batch_manifest.json')).read_text())['jobs']
    assert len(jobs) == 4 * len(METHODS)
    assert set(x['method'] for x in jobs) == set(METHODS)
    assert all(x['level'] == default_level(x['method']) for x in jobs)


def test_llm_export_schema(traces, tmp_path):
    out = tmp_path / 'export'
    main(['--method', 'gemma4', '--path_traces', str(traces), '--run-dir', str(out), '--test_size', '0.33'])
    assert (out / 'data/train.csv').is_file()
    assert (out / 'data/test.csv').is_file()
    assert json.loads((out / 'status.json').read_text())['status'] == 'exported'


def test_sequence_ids_follow_step_order_and_truncation():
    frame = pd.DataFrame({'trace_id': ['b']*102 + ['a'], 'step_num': list(range(102)) + [0], 'row_id': list(range(102)) + [999]})
    assert sequence_row_ids(frame) == list(range(100)) + [999]


def test_missing_interpreter_is_recorded(tmp_path):
    from benchmark_AD.run_all import run_logged
    log = tmp_path / 'missing.log'
    assert run_logged([str(tmp_path / 'missing-python')], log) == 127
    assert 'Could not start' in log.read_text()


def test_batch_real_subprocess_and_summary(traces, tmp_path):
    out = tmp_path / 'batch-out'
    assert run_all(['--methods', 'random', '--datasets', 'EscapeRoom_v3_gsm_hard',
                    '--trace-types', 'tool', '--path-traces', str(traces),
                    '--output-dir', str(out), '--test-size', '0.33']) == 0
    summary = pd.read_csv(next(out.rglob('batch_summary.csv')))
    assert summary.iloc[0]['status'] == 'completed'
    assert Path(summary.iloc[0]['predictions']).is_file()
    assert np.isfinite(summary.iloc[0]['auc'])
