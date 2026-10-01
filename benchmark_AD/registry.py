"""Method dispatch without importing optional ML dependencies."""
METHODS = {
    'random': ('tabular.random', 'run_random'),
    'random_forest': ('tabular.random_forest', 'run_random_forest'),
    'svm': ('tabular.svm', 'run_svm'),
    'xgboost': ('tabular.xgboost', 'run_xgboost'),
    'knn': ('tabular.knn', 'run_knn'),
    'lof': ('tabular.lof', 'run_lof'),
    'isolation_forest': ('tabular.isolation_forest', 'run_isolation_forest'),
    'ocsvm': ('tabular.ocsvm', 'run_ocsvm'),
    'autoencoder': ('tabular.autoencoder', 'run_autoencoder'),
    'deepsvdd': ('tabular.deep_svdd', 'run_deep_svdd'),
    'devnet': ('tabular.devnet', 'run_devnet'),
    'deepsad': ('tabular.deepsad', 'run_deep_sad'),
    'g_safeguard': ('g_safeguard.g_safeguard', 'run_g_safeguard'),
    'g_safeguard_semi_supervised': ('g_safeguard.g_safeguard', 'run_g_safeguard'),
    'blindguard': ('blindguard.blindguard', 'run_blindguard'),
    'tam': ('tam.tam', 'run_tam'),
    'dominant': ('dominant.dominant', 'run_dominant'),
    'ggad': ('ggad.ggad', 'run_ggad'),
    'igad': ('igad.igad', 'run_igad'),
    'rqgnn': ('rqgnn.rqgnn', 'run_rqgnn'),
    'ma2df_improved': ('ma2df.ma2df', 'run_ma2df'),
    'anomaly_transformer': ('anomaly_transformer.anomaly_transformer', 'run_anomaly_transformer'),
    'sup_transformer': ('sup_transformer.sup_transformer', 'run_sup_transformer'),
    'gemma4': ('llm_as_judge.gemma4', 'run_gemma4'),
}
TRACE_ONLY = {'igad', 'rqgnn', 'ma2df_improved'}
ACTION_ONLY = {'anomaly_transformer', 'sup_transformer', 'gemma4', 'ggad'}
DATASETS = ('EscapeRoom_v3_gsm_hard', 'EscapeRoom_v3_live_code_bench', 'EscapeRoom_v2', 'gsm_hard')


def default_level(method):
    return 'trace' if method in TRACE_ONLY else 'action'


def validate_level(method, level):
    if method in TRACE_ONLY and level != 'trace':
        raise ValueError(f'{method} produces graph/trace predictions; use --dataset_level trace')
    if method in ACTION_ONLY and level != 'action':
        raise ValueError(f'{method} supports --dataset_level action only')
