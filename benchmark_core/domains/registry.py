"""Resolve a clue domain without mutating global prompts or loaders."""

from __future__ import annotations

from typing import Dict

from benchmark_core.domains.base import ClueDomain

_DOMAINS: Dict[str, ClueDomain] = {}


def supported_domains():
    return ("gsm-hard", "livecodebench")


def get_domain(name: str) -> ClueDomain:
    normalized = name.lower().replace("_", "-")
    aliases = {"gsmhard": "gsm-hard", "livecode-bench": "livecodebench"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in supported_domains():
        raise ValueError(f"Unknown clue domain '{name}'. Supported: {supported_domains()}")
    if normalized not in _DOMAINS:
        if normalized == "gsm-hard":
            from benchmark_core.domains.gsm_hard.domain import GSMHardDomain
            _DOMAINS[normalized] = GSMHardDomain()
        else:
            from benchmark_core.domains.livecodebench.domain import LiveCodeBenchDomain
            _DOMAINS[normalized] = LiveCodeBenchDomain()
    return _DOMAINS[normalized]
