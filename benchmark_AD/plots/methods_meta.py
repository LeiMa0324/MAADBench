"""Shared method metadata for AD plots.

This mirrors the paper_resources/plots/exp_analysis method ordering so figures
stay visually consistent.
"""
from __future__ import annotations


LLM_CSV_SUFFIX: dict[str, str] = {
    "gemma4_e2b_zs": "gemma-4-E2B_zero_shot",
    "gemma4_e4b_zs": "gemma-4-E4B_zero_shot",
    "gemma4_31b_zs": "gemma-4-31B_zero_shot",
    "gemma4_26b_a4b_zs": "gemma-4-26B-A4B_zero_shot",
    "gemma4_e2b_fs": "gemma-4-E2B_few_shot",
    "gemma4_e4b_fs": "gemma-4-E4B_few_shot",
    "gemma4_31b_fs": "gemma-4-31B_few_shot",
    "gemma4_26b_a4b_fs": "gemma-4-26B-A4B_few_shot",
}

METHOD_SHORT: dict[str, str] = {
    "autoencoder_clean": "AutoEncoder",
    "deepsvdd_clean": "DeepSVDD",
    "isolation_forest": "iForest",
    "knn": "kNN",
    "lof": "LOF",
    "ocsvm_clean": "OC-SVM",
    "random_forest": "Rand.Forest",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "blindguard_clean": "BlindGuard",
    "dominant_clean": "DOMINANT",
    "tam_clean": "TAM",
    "g_safeguard": "G-Safeguard",
    "deepsad": "DeepSAD",
    "devnet": "DevNet",
    "ggad": "GGAD",
    "g_safeguard_semi_supervised": "G-Safeguard",
    "gemma4_e2b_zs": "Gemma4-E2B",
    "gemma4_e4b_zs": "Gemma4-E4B",
    "gemma4_31b_zs": "Gemma4-31B",
    "gemma4_26b_a4b_zs": "Gemma4-26B",
    "gemma4_e2b_fs": "Gemma4-E2B",
    "gemma4_e4b_fs": "Gemma4-E4B",
    "gemma4_31b_fs": "Gemma4-31B",
    "gemma4_26b_a4b_fs": "Gemma4-26B",
}

METHOD_SHORTEST = METHOD_SHORT

METHOD_CATEGORY: dict[str, str] = {
    "isolation_forest": "Unsupervised",
    "knn": "Unsupervised",
    "lof": "Unsupervised",
    "gemma4_e2b_zs": "Unsupervised",
    "gemma4_e4b_zs": "Unsupervised",
    "gemma4_31b_zs": "Unsupervised",
    "gemma4_26b_a4b_zs": "Unsupervised",
    "blindguard_clean": "OCC",
    "ocsvm_clean": "OCC",
    "autoencoder_clean": "OCC",
    "deepsvdd_clean": "OCC",
    "tam_clean": "OCC",
    "dominant_clean": "OCC",
    "deepsad": "Semi-supervised",
    "devnet": "Semi-supervised",
    "ggad": "Semi-supervised",
    "g_safeguard_semi_supervised": "Semi-supervised",
    "gemma4_e2b_fs": "Semi-supervised",
    "gemma4_e4b_fs": "Semi-supervised",
    "gemma4_31b_fs": "Semi-supervised",
    "gemma4_26b_a4b_fs": "Semi-supervised",
    "g_safeguard": "Supervised",
    "svm": "Supervised",
    "random_forest": "Supervised",
    "xgboost": "Supervised",
}

CATEGORY_ORDER: list[str] = [
    "Unsupervised",
    "OCC",
    "Semi-supervised",
    "Supervised",
]

METHOD_MODALITY: dict[str, str] = {
    "isolation_forest": "Tabular",
    "knn": "Tabular",
    "lof": "Tabular",
    "ocsvm_clean": "Tabular",
    "autoencoder_clean": "Tabular",
    "deepsvdd_clean": "Tabular",
    "svm": "Tabular",
    "random_forest": "Tabular",
    "xgboost": "Tabular",
    "dominant_clean": "Graph",
    "tam_clean": "Graph",
    "ggad": "Graph",
    "blindguard_clean": "MAS-specific",
    "g_safeguard": "MAS-specific",
    "g_safeguard_semi_supervised": "MAS-specific",
    "deepsad": "Tabular",
    "devnet": "Tabular",
    "gemma4_e2b_zs": "LLM",
    "gemma4_e4b_zs": "LLM",
    "gemma4_26b_a4b_zs": "LLM",
    "gemma4_31b_zs": "LLM",
    "gemma4_e2b_fs": "LLM",
    "gemma4_e4b_fs": "LLM",
    "gemma4_31b_fs": "LLM",
    "gemma4_26b_a4b_fs": "LLM",
}

MODALITY_ORDER: list[str] = ["Tabular", "Graph", "MAS-specific", "LLM"]

METHOD_ORDER_IN_CATEGORY: dict[str, list[str]] = {
    "Unsupervised": [
        "isolation_forest", "knn", "lof",
        "gemma4_e2b_zs", "gemma4_e4b_zs",
        "gemma4_26b_a4b_zs", "gemma4_31b_zs",
    ],
    "OCC": [
        "autoencoder_clean", "deepsvdd_clean",
        "ocsvm_clean",
        "dominant_clean", "tam_clean",
        "blindguard_clean",
    ],
    "Semi-supervised": [
        "deepsad", "devnet",
        "ggad",
        "g_safeguard_semi_supervised",
        "gemma4_e2b_fs", "gemma4_e4b_fs",
        "gemma4_26b_a4b_fs", "gemma4_31b_fs",
    ],
    "Supervised": [
        "svm", "random_forest", "xgboost",
        "g_safeguard",
    ],
}

METHOD_ORDER: list[str] = [
    method
    for category in CATEGORY_ORDER
    for method in METHOD_ORDER_IN_CATEGORY[category]
]

BACKBONE_ORDER: list[str] = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "deepseek-reasoner",
    "gpt-4.1",
    "gpt-5.4",
]

BACKBONE_DISPLAY: dict[str, str] = {
    "claude-opus-4-8": "Claude\nOpus 4.8",
    "claude-sonnet-4-6": "Claude\nSonnet 4.6",
    "deepseek-reasoner": "DeepSeek-R1",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.4": "GPT-5.4",
}


def ordered_methods(available: list[str] | set[str]) -> list[str]:
    avail = set(available)
    ordered = [method for method in METHOD_ORDER if method in avail]
    ordered.extend(sorted(avail - set(ordered)))
    return ordered


def csv_score_col(method_key: str) -> str:
    return f"pred_score_{LLM_CSV_SUFFIX.get(method_key, method_key)}"


def csv_label_col(method_key: str) -> str:
    return f"pred_label_{LLM_CSV_SUFFIX.get(method_key, method_key)}"
