"""Small real-model integration runs; embeddings are stubbed to avoid downloads.

These validate method dispatch, shapes, IDs and artifacts, not model quality.
Run with the optional ML dependencies installed.
"""
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark_AD.main import main
from benchmark_AD.tests.test_pipeline import traces


class LocalEmbedding:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.stack([self.encode(x, **kwargs) for x in text])
        seed = int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=384).astype(np.float32)


@pytest.mark.parametrize('method', [
    'random_forest', 'svm', 'xgboost', 'knn', 'lof', 'isolation_forest', 'ocsvm',
    'autoencoder', 'deepsvdd', 'deepsad', 'devnet', 'g_safeguard',
    'g_safeguard_semi_supervised', 'blindguard', 'tam', 'dominant',
    'ggad', 'igad', 'rqgnn', 'ma2df_improved', 'anomaly_transformer', 'sup_transformer',
])
def test_model_pipeline(method, traces, tmp_path, monkeypatch):
    if method == 'devnet':
        pytest.importorskip('tensorflow')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_geometric')
    st = pytest.importorskip('sentence_transformers')
    pytest.importorskip('pyod')
    if method == 'xgboost':
        try:
            importlib.import_module('xgboost')
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            pytest.skip(f'XGBoost runtime unavailable: {exc}')
    torch.set_num_threads(1)
    monkeypatch.setattr(st, 'SentenceTransformer', LocalEmbedding)
    # Some modules may already have bound the imported class in an earlier case.
    for module_name in [
        'tabular.prepare_traces_tabular', 'g_safeguard.gen_training_dataset',
        'g_safeguard.convert_lomas_traces', 'anomaly_transformer.convert_lomas_traces',
        'sup_transformer.sup_transformer',
    ]:
        module = importlib.import_module('benchmark_AD.methods.' + module_name)
        if hasattr(module, 'SentenceTransformer'):
            monkeypatch.setattr(module, 'SentenceTransformer', LocalEmbedding)
    out = tmp_path / method
    main(['--method', method, '--path_traces', str(traces), '--run-dir', str(out),
          '--test_size', '0.33', '--epochs', '1', '--perc_outliers_train', '0.5'])
    predictions = pd.read_csv(out / 'predictions.csv')
    assert predictions.test_id.is_unique
    assert np.isfinite(predictions.pred_score).all()
    assert len(predictions) == len(pd.read_csv(out / 'test_manifest.csv'))


def test_empty_trace_graph_encoding():
    pytest.importorskip('sentence_transformers')
    from benchmark_AD.methods.g_safeguard.convert_lomas_traces import create_dataset_trace_level
    frame = pd.DataFrame([{'label': 1, 'source_file': 'empty.json',
                           'trace': {'trace': [], 'config': {'agents': []}}}])
    graphs = create_dataset_trace_level(frame)
    assert graphs[0]['adj_matrix'] == [[1]]
    assert graphs[0]['attacker_idxes'] == [0]


def test_trace_tabular_occ_with_empty_trace(traces, tmp_path, monkeypatch):
    pytest.importorskip('sentence_transformers')
    from benchmark_AD.methods.tabular import prepare_traces_tabular as tabular
    from benchmark_AD.utils import setup_datasets
    from benchmark_AD.tests.test_pipeline import args
    monkeypatch.setattr(tabular, 'SentenceTransformer', LocalEmbedding)
    config = args(traces, dataset_level='trace', run_dir=tmp_path, occ_training='clean')
    train, test = setup_datasets(config)
    train.at[0, 'trace'] = {'trace': []}
    X_train, y_train, X_test, y_test, _, _ = tabular.prepare_traces_tabular(config, 'occ', train, test)
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    assert not y_train.any()
    columns = __import__('json').loads((tmp_path / 'data/feature_columns.json').read_text())
    assert not {'label', 'task', 'row_id', 'source_file', 'room_group'} & set(columns)
