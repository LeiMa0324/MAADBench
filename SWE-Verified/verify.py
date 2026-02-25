"""
main.py — Entry point for evaluating a MetaGPT-style MAS on SWE-bench Verified.

Usage:
    # From project root:
    PYTHONPATH=. python SWE-Verified/main.py

    # With options:
    PYTHONPATH=. python SWE-Verified/main.py --sample_size 10 --difficulty easy
"""

import sys
import os
from pathlib import Path
from utils import *

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Load .env and bootstrap ~/.metagpt/config2.yaml BEFORE metagpt imports ──
# MetaGPT calls Config.default() at module level, so the config file must
# exist before any metagpt import is triggered.
from dotenv import load_dotenv
import yaml as _yaml

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "configs/config.yaml")) as _f:
    _proj_cfg = _yaml.safe_load(_f)

_llm = _proj_cfg.get("llm", {})
_metagpt_cfg = Path.home() / ".metagpt" / "config2.yaml"
_metagpt_cfg.parent.mkdir(exist_ok=True)
_metagpt_cfg.write_text(
    f"llm:\n"
    f"  api_type: \"{_llm.get('api_type', 'openai')}\"\n"
    f"  model: \"{_llm.get('model', 'gpt-4.1-mini')}\"\n"
    f"  api_key: \"{os.environ.get('OPENAI_API_KEY', '')}\"\n"
    f"  temperature: {_llm.get('temperature', 0.0)}\n"
)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import yaml
import wandb

from swe_dataloader import SWEDataLoader
from swe_mas import SWEOrchestrator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mas_framework.mas import Agent  # noqa (ensures mas_framework importable)


DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MetaGPT MAS on SWE-bench Verified.")
    parser.add_argument("--dataset", type=str, default="SWE-Verified")
    parser.add_argument("--mas_arch", type=str, default="MetaGPT")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml")
    parser.add_argument("--data_file", type=str, default="swe_bench_verified.jsonl")
    parser.add_argument("--sample_size", type=int, default=-1,
                        help="Total instances to sample (-1 = all).")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter by difficulty: easy / medium / hard / expert. "
                             "Default: run all difficulties.")
    parser.add_argument("--output_dir", type=str, default="traces")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



if __name__ == "__main__":

    # ── Resolve paths relative to this file ────────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))

    predictions_dir = os.path.join(here, "predictions")

    pred_path = os.path.join(predictions_dir, "predictions.jsonl")

    with open(pred_path, "r", encoding="utf-8") as f:
        all_predictions = [json.loads(line) for line in f if line.strip()]

    instance_ids = [p["instance_id"] for p in all_predictions]
    print(f"Loaded {len(instance_ids)} predictions from {pred_path}")

    # 所有 task 跑完后，统一跑一次 harness
    from swebench.harness.run_evaluation import main as harness_main  # type: ignore[import-untyped]

    print("========= Prediction Harness ==========")
    harness_main(
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        instance_ids=instance_ids,
        predictions_path=pred_path,
        max_workers=8,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id="x",
        timeout=1800,
        namespace=None,
        rewrite_reports=False,
        modal=False,
    )
