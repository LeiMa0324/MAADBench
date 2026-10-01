"""
mast_taxonomy.py — MAST Failure Mode Taxonomy constants.

Source: "MAST: A Multi-Agent System Failure Taxonomy" (2025)
14 failure modes across 3 failure categories (FC).
"""

# ---------------------------------------------------------------------------
# Full taxonomy: FM_ID -> metadata
# ---------------------------------------------------------------------------

MAST_TAXONOMY: dict = {

    # ── FC1: System Design Issues ──────────────────────────────────────────
    # Failures originating from system design decisions and poor/ambiguous
    # prompt specifications.

    "FM-1.1": {
        "category":    "FC1",
        "name":        "Disobey task specification",
        "description": (
            "Failure to adhere to the specified constraints or requirements "
            "of a given task, leading to suboptimal or incorrect outcomes."
        ),
        "prevalence":  0.118,   # 11.8% in MAST-Data
    },
    "FM-1.2": {
        "category":    "FC1",
        "name":        "Disobey role specification",
        "description": (
            "Failure to adhere to the defined responsibilities and constraints "
            "of an assigned role, potentially leading to an agent behaving "
            "like another."
        ),
        "prevalence":  0.015,   # 1.5%
    },
    "FM-1.3": {
        "category":    "FC1",
        "name":        "Step repetition",
        "description": (
            "Unnecessary reiteration of previously completed steps in a "
            "process, potentially causing delays or errors in task completion."
        ),
        "prevalence":  0.157,   # 15.7%
    },
    "FM-1.4": {
        "category":    "FC1",
        "name":        "Loss of conversation history",
        "description": (
            "Unexpected context truncation, disregarding recent interaction "
            "history and reverting to an antecedent conversational state."
        ),
        "prevalence":  0.028,   # 2.8%
    },
    "FM-1.5": {
        "category":    "FC1",
        "name":        "Unaware of termination conditions",
        "description": (
            "Lack of recognition or understanding of the criteria that should "
            "trigger the termination of the agents' interaction, potentially "
            "leading to unnecessary continuation."
        ),
        "prevalence":  0.124,   # 12.4%
    },

    # ── FC2: Inter-Agent Misalignment ──────────────────────────────────────
    # Failures arising from ineffective communication, poor collaboration,
    # conflicting behaviors among agents, and gradual derailment.

    "FM-2.1": {
        "category":    "FC2",
        "name":        "Conversation reset",
        "description": (
            "Unexpected or unwarranted restarting of a dialogue, potentially "
            "losing context and progress made in the interaction."
        ),
        "prevalence":  0.022,   # 2.2%
    },
    "FM-2.2": {
        "category":    "FC2",
        "name":        "Fail to ask for clarification",
        "description": (
            "Inability to request additional information when faced with "
            "unclear or incomplete data, potentially resulting in incorrect "
            "actions."
        ),
        "prevalence":  0.068,   # 6.8%
    },
    "FM-2.3": {
        "category":    "FC2",
        "name":        "Task derailment",
        "description": (
            "Deviation from the intended objective or focus of a given task, "
            "potentially resulting in irrelevant or unproductive actions."
        ),
        "prevalence":  0.074,   # 7.4%
    },
    "FM-2.4": {
        "category":    "FC2",
        "name":        "Information withholding",
        "description": (
            "Failure to share or communicate important data or insights that "
            "an agent possesses and could impact decision-making of other "
            "agents if shared."
        ),
        "prevalence":  0.0085,  # 0.85%
    },
    "FM-2.5": {
        "category":    "FC2",
        "name":        "Ignored other agent's input",
        "description": (
            "Disregarding or failing to adequately consider input or "
            "recommendations provided by other agents in the system, "
            "potentially leading to suboptimal decisions or missed "
            "opportunities for collaboration."
        ),
        "prevalence":  0.019,   # 1.9%
    },
    "FM-2.6": {
        "category":    "FC2",
        "name":        "Reasoning-action mismatch",
        "description": (
            "Discrepancy between the logical reasoning process and the actual "
            "actions taken by the agent, potentially resulting in unexpected "
            "or undesired behaviors."
        ),
        "prevalence":  0.132,   # 13.2%
    },

    # ── FC3: Task Verification ─────────────────────────────────────────────
    # Failures from premature termination and insufficient mechanisms to
    # guarantee accuracy, completeness, and reliability of outcomes.

    "FM-3.1": {
        "category":    "FC3",
        "name":        "Premature termination",
        "description": (
            "Ending a dialogue, interaction or task before all necessary "
            "information has been exchanged or objectives have been met, "
            "potentially resulting in incomplete or incorrect outcomes."
        ),
        "prevalence":  0.062,   # 6.2%
    },
    "FM-3.2": {
        "category":    "FC3",
        "name":        "No or incomplete verification",
        "description": (
            "(Partial) omission of proper checking or confirmation of task "
            "outcomes or system outputs, potentially allowing errors or "
            "inconsistencies to propagate undetected."
        ),
        "prevalence":  0.082,   # 8.2%
    },
    "FM-3.3": {
        "category":    "FC3",
        "name":        "Incorrect verification",
        "description": (
            "Failure to adequately validate or cross-check crucial information "
            "or decisions during the iterations, potentially leading to errors "
            "or vulnerabilities in the system."
        ),
        "prevalence":  0.091,   # 9.1%
    },
}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

# All FM IDs grouped by failure category
FC1_FMS = [k for k, v in MAST_TAXONOMY.items() if v["category"] == "FC1"]
FC2_FMS = [k for k, v in MAST_TAXONOMY.items() if v["category"] == "FC2"]
FC3_FMS = [k for k, v in MAST_TAXONOMY.items() if v["category"] == "FC3"]

# Flat name lookup: "FM-2.4" -> "Information withholding"
FM_NAMES: dict[str, str] = {k: v["name"] for k, v in MAST_TAXONOMY.items()}

# Short label for display: "FM-2.4: Information withholding"
FM_LABELS: dict[str, str] = {k: f"{k}: {v['name']}" for k, v in MAST_TAXONOMY.items()}

