"""
sage_anomaly.py — Phase 3: Agent-level anomaly injection.

Anomaly injection operates on AgentOutput.content dicts during MAS execution.
The orchestrator calls inject() after an agent produces output, corrupting
the dict before downstream agents see it.

This separation means:
  - The same clean task can be run under multiple anomaly configs
  - Ground truth and trap_log remain untouched
  - Anomaly type and injection step are fully logged for analysis

Supported anomaly types (mapped to MAST failure modes):
  FM-1.1  parser_wrong_structure    Parser returns incorrect tree
  FM-1.3  transformer_repeat        Transformer repeats previous output
  FM-2.4  transformer_withhold      Transformer returns empty/truncated output
  FM-2.5  ignored_input             Transformer ignores rule_selector's instruction
  FM-2.6  rule_action_mismatch      Rule Selector announces rule X but targets wrong node
  FM-3.1  premature_termination     Rule Selector returns NONE too early
  FM-3.2  checker_incomplete        Checker tests only one value
  FM-3.3  checker_wrong_verdict     Checker flips EQUIVALENT / NOT_EQUIVALENT
"""

import copy
import random
from dataclasses import dataclass
from typing import Any, Optional, Callable


# ---------------------------------------------------------------------------
# AnomalyConfig
# ---------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    anomaly_type: str              # one of the FM codes above
    target_agent: str              # "parser" | "rule_selector" | "transformer" | "checker"
    injection_step: int = 0        # which simplification step to corrupt (0-indexed)

    # Registry of all supported types
    SUPPORTED = {
        # "FM-1.1":  "parser_wrong_structure",
        # "FM-1.3":  "transformer_repeat",
        "FM-2.4":  "transformer_withhold",
        "FM-2.5":  "ignored_input",
        "FM-2.6":  "rule_action_mismatch",
        "FM-3.1":  "premature_termination",
        "FM-3.2":  "checker_incomplete",
        "FM-3.3":  "checker_wrong_verdict",
    }

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyConfig":
        """Reconstruct an AnomalyConfig from a task's anomaly dict."""
        return cls(
            anomaly_type=d["anomaly_type"],
            target_agent=d["target_agent"],
            injection_step=d.get("injection_step", 0),
        )


# ---------------------------------------------------------------------------
# AnomalyRecord — attached to evaluation output, not to the task
# ---------------------------------------------------------------------------

@dataclass
class AnomalyRecord:
    anomaly_type: str
    target_agent: str
    injection_step: int
    description: str               # human-readable explanation of what was corrupted
    original_output: Any           # clean content dict
    corrupted_output: Any          # corrupted content dict


# ---------------------------------------------------------------------------
# Individual injectors
# Each injector receives an AgentOutput.content dict and returns a corrupted copy.
# ---------------------------------------------------------------------------

def inject_FM_1_1(content: dict, rng: random.Random) -> dict:
    """
    FM-1.1: Parser disobeys task specification.
    Clear tree and flagged_nodes entirely.
    """
    corrupted = copy.deepcopy(content)
    corrupted["tree"] = {}
    corrupted["flagged_nodes"] = []
    return corrupted


def inject_FM_1_3(
    content: dict,
    previous_content: Optional[dict],
    rng: random.Random,
) -> dict:
    """
    FM-1.3: Step repetition — Transformer returns the same output as the previous step.
    If no previous output is available, returns the current output unchanged (identity).
    """
    if previous_content is not None:
        return copy.deepcopy(previous_content)
    return copy.deepcopy(content)


def inject_FM_2_4(content: dict, rng: random.Random) -> dict:
    """
    FM-2.4: Information withholding — Transformer truncates its output.
    Empty result_expression.
    """
    corrupted = copy.deepcopy(content)
    corrupted["result_expression"] = ""
    return corrupted


def inject_FM_2_5(content: dict, rng: random.Random) -> dict:
    """
    FM-2.5: Ignored other agent's input — Transformer ignores rule_selector's instruction.
    Changes rule_applied AND corrupts result_expression (adds no-op wrapper).
    """
    corrupted = copy.deepcopy(content)
    corrupted["rule_applied"] = "unrelated_rule"
    expr = corrupted.get("result_expression", "x")
    corrupted["result_expression"] = f"({expr} + 0)"
    corrupted.pop("changed_node", None)
    corrupted.pop("changed_node_before", None)
    corrupted.pop("changed_node_after", None)
    return corrupted


def inject_FM_2_6(content: dict, rng: random.Random) -> dict:
    """
    FM-2.6: Reasoning-Action Mismatch.
    Keep selected_rule but point target_node to a wrong node.
    """
    corrupted = copy.deepcopy(content)
    corrupted["target_node"] = "(x / y)"
    return corrupted


def inject_FM_3_1(content: dict, rng: random.Random) -> dict:
    """
    FM-3.1: Premature termination — Rule selector returns NONE too early.
    Sets selected_rule to NONE and clears target/expected fields.
    """
    corrupted = copy.deepcopy(content)
    corrupted["selected_rule"] = "NONE"
    corrupted["target_node"] = ""
    corrupted["expected_result"] = ""
    corrupted.pop("reason", None)
    return corrupted


def inject_FM_3_2(content: dict, rng: random.Random) -> dict:
    """
    FM-3.2: Incomplete verification — Checker tests fewer values than required.
    Keep only one test case.
    """
    corrupted = copy.deepcopy(content)
    test_cases = corrupted.get("test_cases", [])
    if len(test_cases) > 1:
        corrupted["test_cases"] = test_cases[:1]
    return corrupted


def inject_FM_3_3(content: dict, rng: random.Random) -> dict:
    """
    FM-3.3: Incorrect verification — Checker flips its verdict.
    """
    corrupted = copy.deepcopy(content)
    verdict = corrupted.get("verdict", "")
    if verdict == "EQUIVALENT":
        corrupted["verdict"] = "NOT_EQUIVALENT"
    elif verdict == "NOT_EQUIVALENT":
        corrupted["verdict"] = "EQUIVALENT"
    corrupted["reason"] = "verification passed (all test cases matched)"
    return corrupted


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_INJECTOR_MAP: dict[str, Callable] = {
    "FM-1.1":  inject_FM_1_1,
    "FM-1.3":  inject_FM_1_3,
    "FM-2.4":  inject_FM_2_4,
    "FM-2.5":  inject_FM_2_5,
    "FM-2.6":  inject_FM_2_6,
    "FM-3.1":  inject_FM_3_1,
    "FM-3.2":  inject_FM_3_2,
    "FM-3.3":  inject_FM_3_3,
}

_AGENT_MAP = {
    "FM-1.1":  "parser",
    "FM-1.3":  "transformer",
    "FM-2.4":  "transformer",
    "FM-2.5":  "transformer",
    "FM-2.6":  "rule_selector",
    "FM-3.1":  "rule_selector",
    "FM-3.2":  "checker",
    "FM-3.3":  "checker",
}

_DESCRIPTION_MAP = {
    "FM-1.1":  "Parser omits tree structure (disobeys task specification)",
    "FM-1.3":  "Transformer repeats previous step output (step repetition)",
    "FM-2.4":  "Transformer withholds result expression (information withholding)",
    "FM-2.5":  "Transformer ignores rule_selector instruction (ignored input)",
    "FM-2.6":  "Rule Selector targets wrong node (reasoning-action mismatch)",
    "FM-3.1":  "Rule Selector returns NONE too early (premature termination)",
    "FM-3.2":  "Checker uses insufficient test cases (incomplete verification)",
    "FM-3.3":  "Checker flips EQUIVALENT/NOT_EQUIVALENT verdict (incorrect verification)",
}


# ---------------------------------------------------------------------------
# AnomalyInjector — unified interface
# ---------------------------------------------------------------------------

class AnomalyInjector:
    """
    Injector that corrupts AgentOutput.content dicts.

    Usage:
        injector = AnomalyInjector(seed=42)
        config = injector.pick_random_config(task)
        # ... pass config to orchestrator via set_anomaly()
        # orchestrator calls inject() after the targeted agent executes
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def pick_random_config(self, task: dict, anomaly_type: Optional[str] = None) -> AnomalyConfig:
        """Pick config with given FM type (or random). Step is randomly chosen."""
        if anomaly_type is None:
            types = list(AnomalyConfig.SUPPORTED.keys())
            anomaly_type = self.rng.choice(types)
        elif anomaly_type not in _INJECTOR_MAP:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}. "
                             f"Supported: {list(_INJECTOR_MAP.keys())}")
        target_agent = _AGENT_MAP[anomaly_type]

        k = task.get("k_steps", 1)
        if target_agent in ("parser", "checker"):
            injection_step = 0
        elif anomaly_type == "FM-3.1":
            # Premature termination must happen early to be meaningful
            injection_step = self.rng.randint(0, max(0, k // 2 - 1))
        else:
            injection_step = self.rng.randint(0, max(0, k - 1))

        return AnomalyConfig(
            anomaly_type=anomaly_type,
            target_agent=target_agent,
            injection_step=injection_step,
        )

    def inject(
        self,
        anomaly_type: str,
        agent_output: dict,
        previous_output: Optional[dict] = None,
        injection_step: int = 0,
    ) -> tuple[dict, AnomalyRecord]:
        """
        Corrupt an agent's content dict.

        Returns (corrupted_content_dict, AnomalyRecord).
        """
        if anomaly_type not in _INJECTOR_MAP:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}. "
                             f"Supported: {list(_INJECTOR_MAP.keys())}")

        fn = _INJECTOR_MAP[anomaly_type]

        # FM-1.3 needs previous_output as extra arg
        if anomaly_type == "FM-1.3":
            corrupted = fn(agent_output, previous_output, self.rng)
        else:
            corrupted = fn(agent_output, self.rng)

        record = AnomalyRecord(
            anomaly_type=anomaly_type,
            target_agent=_AGENT_MAP[anomaly_type],
            injection_step=injection_step,
            description=_DESCRIPTION_MAP[anomaly_type],
            original_output=copy.deepcopy(agent_output),
            corrupted_output=corrupted,
        )
        return corrupted, record
