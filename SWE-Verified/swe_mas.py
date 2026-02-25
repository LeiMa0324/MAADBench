"""
swe_mas.py — MetaGPT-based MAS orchestrator for SWE-bench Verified.

Actions and agents are fully defined in config.yaml:
  - actions: defines name + prompt for each action
  - agents:  references actions by name, declares watch list and memories_k
"""
import subprocess
import tempfile
import shutil
import re
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Type
from utils import *

import yaml
from metagpt.actions import Action, UserRequirement
from metagpt.config2 import Config, LLMConfig
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
from pydantic import PrivateAttr


# ── Actions ───────────────────────────────────────────────────────────────────
# Distinct classes are required for MetaGPT's _watch() message-bus routing.
# Prompts are injected at construction time via prompt_template.

class AnalyzeIssue(Action):
    name: str = "AnalyzeIssue"
    prompt_template: str = ""
    async def run(self, context: str) -> str:
        return await self._aask(f"{self.prompt_template}\n\n{context}")

class PlanSolution(Action):
    name: str = "PlanSolution"
    prompt_template: str = ""
    async def run(self, context: str) -> str:
        return await self._aask(f"{self.prompt_template}\n\n{context}")

class WritePatch(Action):
    name: str = "WritePatch"
    prompt_template: str = ""
    async def run(self, context: str) -> str:
        return await self._aask(f"{self.prompt_template}\n\n{context}")

class ReviewPatch(Action):
    name: str = "ReviewPatch"
    prompt_template: str = ""
    async def run(self, context: str) -> str:
        return await self._aask(f"{self.prompt_template}\n\n{context}")


# Maps action name (from config) → Action class
ACTION_MAP: Dict[str, Type[Action]] = {
    "AnalyzeIssue": AnalyzeIssue,
    "PlanSolution": PlanSolution,
    "WritePatch":   WritePatch,
    "ReviewPatch":  ReviewPatch,
}


# ── Generic Role ──────────────────────────────────────────────────────────────

class SWERole(Role):
    """
    One configurable Role class covers all agents.
    trace_sink is a shared list owned by SWEOrchestrator; messages are appended
    directly in _act so we never rely on rc.memory (which only stores incoming
    messages, not the role's own output).
    """

    # PrivateAttr is required for underscore-prefixed attrs on Pydantic v2 models.
    _memories_k: int  = PrivateAttr(default=3)
    _trace_sink: list = PrivateAttr(default_factory=list)

    def __init__(
        self,
        action_cls: Type[Action],
        watch_classes: List[Type[Action]],
        prompt: str = "",
        memories_k: int = 3,
        trace_sink: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_actions([action_cls(prompt_template=prompt)])
        self._watch(watch_classes if watch_classes else [UserRequirement])
        self._memories_k = memories_k
        self._trace_sink = trace_sink if trace_sink is not None else []

    async def _act(self) -> Message:
        todo = self.rc.todo
        context = "\n\n---\n\n".join(m.content for m in self.get_memories(k=self._memories_k))
        result = await todo.run(context)
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self._trace_sink.append(msg)   # capture output at the source
        return msg


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SWEOrchestrator:
    """Wraps a MetaGPT Team so main.py can call it synchronously."""

    def __init__(
        self,
        metagpt_config: Config,
        agents_config: list,
        action_prompts: Dict[str, str],
    ):
        self._config = metagpt_config
        self._agents = agents_config           # ordered list of agent dicts from config
        self._action_prompts = action_prompts  # action_name -> prompt string
        self.trace: list = []
        self.prediction = {}

    # ── Public interface ────────────────────────────────────────────────────

    def run(self, problem: str, instance_id, context: Optional[str] = None) -> dict:
        return asyncio.run(self._arun(problem, instance_id, context))

    def save_execution_trace(
        self,
        config,
        question_data: dict,
        output_dir: str = "traces",
        trace_id: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        if trace_id is None:
            trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        payload = {"config": config, "question": question_data, "trace": self.trace, "prediction": self.prediction}
        path = os.path.join(output_dir, f"{trace_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_config(cls, config_path: str) -> "SWEOrchestrator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        llm = cfg.get("llm", {})
        llm_kwargs = dict(
            api_type=llm.get("api_type", "openai"),
            model=llm.get("model", ""),
            api_key=os.environ.get("OPENAI_API_KEY") or llm.get("api_key", ""),
            temperature=llm.get("temperature", 0.0),
            max_token=llm.get("max_new_tokens", 4096),
        )
        if llm.get("base_url"):
            llm_kwargs["base_url"] = llm["base_url"]
        metagpt_config = Config(llm=LLMConfig(**llm_kwargs))
        action_prompts = {a["name"]: a.get("prompt", "") for a in cfg.get("actions", [])}
        agents_config  = cfg.get("agents", [])

        return cls(
            metagpt_config=metagpt_config,
            agents_config=agents_config,
            action_prompts=action_prompts,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_role(self, agent_cfg: dict, trace_sink: Optional[list] = None) -> SWERole:
        action_name   = agent_cfg["action"]
        action_cls    = ACTION_MAP[action_name]
        watch_names   = agent_cfg.get("watches", [])
        watch_classes = [ACTION_MAP[w] for w in watch_names] if watch_names else []

        return SWERole(
            action_cls=action_cls,
            watch_classes=watch_classes,
            prompt=self._action_prompts.get(action_name, ""),
            memories_k=agent_cfg.get("memories_k", 3),
            trace_sink=trace_sink,
            name=agent_cfg.get("name", agent_cfg["role"]),
            profile=agent_cfg.get("profile", agent_cfg["role"]),
            goal=agent_cfg.get("goal", ""),
            constraints=agent_cfg.get("constraints", ""),
            config=self._config,
        )

    # ── Async core ──────────────────────────────────────────────────────────

    async def _arun(self, problem: str, instance_id, context: Optional[str] = None) -> dict:
        self.trace = []
        ctx = json.loads(context) if context else {}

        # ── Retrieve actual source code ───────────────────────────────────────
        repo = ctx.get("repo", "")
        base_commit = ctx.get("base_commit", "")
        code_context = ""
        if repo and base_commit:
            print(f"  [CodeRetriever] Fetching relevant files from {repo}@{base_commit[:8]}...")
            code_context = self._retrieve_relevant_code(repo, base_commit, problem)
            print(f"  [CodeRetriever] Retrieved {len(code_context)} chars")

        # ── Build initial message ─────────────────────────────────────────────
        initial_message = (
            f"Repository: {repo}  (version {ctx.get('version', '')})\n\n"
            f"Problem statement:\n{problem}\n\n"
            f"Hints:\n{ctx.get('hints_text', '(none)')}\n\n"
            f"Tests that must FAIL→PASS after fix:\n"
            f"{json.dumps(ctx.get('fail_to_pass', []), indent=2)}\n\n"
            f"Tests that must remain PASS→PASS (no regression):\n"
            f"{json.dumps(ctx.get('pass_to_pass', []), indent=2)}\n\n"
            f"RELEVANT SOURCE CODE (use this as ground truth for writing the patch):\n"
            f"{code_context}"
        )


        sink: List[Message] = []
        team = Team(config=self._config)
        team.hire([self._build_role(a, trace_sink=sink) for a in self._agents])
        team.run_project(initial_message)
        await team.run(n_round=len(self._agents))

        # sink is populated directly by SWERole._act — no memory fishing needed.
        final_answer = None
        for msg in sink:
            print(f"  =========[{msg.role}]=========")
            cause = str(getattr(msg, "cause_by", ""))
            self.trace.append({
                "agent": msg.role,
                "cause_by": cause,
                "output": {"content": msg.content},
            })

            parsed = extract_json(msg.content)
            if parsed and "final_answer" in parsed:
                fa = parsed["final_answer"]
                if fa:  # 非空才更新
                    final_answer = fa
        self.prediction= self.format_prediction(instance_id, final_answer)
        return self.prediction

    def format_prediction(self, instance_id: str, final_patch: str, model_name: str = "metagpt_mas") -> dict:
        return {
            "instance_id": instance_id,
            "model_patch": final_patch,
            "model_name_or_path": model_name
        }

    def _retrieve_relevant_code(
            self,
            repo: str,
            base_commit: str,
            problem_statement: str,
            max_files: int = 3,
            max_lines_per_file: int = 150,
    ) -> str:
        """
        Clone repo at base_commit, find files most relevant to the issue,
        return their content as a formatted string.
        """
        work_dir = None
        try:
            work_dir = tempfile.mkdtemp(prefix="swe_ctx_")
            repo_url = f"https://github.com/{repo}.git"

            # Shallow clone — much faster than full clone
            subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", repo_url, work_dir],
                check=True, capture_output=True, timeout=120,
            )
            subprocess.run(
                ["git", "-C", work_dir, "checkout", "--quiet", base_commit],
                capture_output=True, timeout=30,
            )

            # ── Find relevant files via keyword grep ──────────────────────────
            # Extract candidate keywords from problem statement
            # (class names, function names: CamelCase or snake_case identifiers)
            keywords = re.findall(r'\b([A-Z][a-zA-Z]{2,}|[a-z_]{3,})\b', problem_statement)
            keywords = list(dict.fromkeys(keywords))[:10]  # deduplicate, top 10

            candidate_files: dict[str, int] = {}  # file -> hit count
            for kw in keywords:
                r = subprocess.run(
                    ["git", "-C", work_dir, "grep", "-rl", "--include=*.py", kw],
                    capture_output=True, text=True, timeout=15,
                )
                for fpath in r.stdout.strip().splitlines():
                    candidate_files[fpath] = candidate_files.get(fpath, 0) + 1

            # Sort by hit count, take top N
            top_files = sorted(candidate_files, key=lambda f: -candidate_files[f])[:max_files]

            if not top_files:
                return "(No relevant files found via keyword search)"

            # ── Read file contents ────────────────────────────────────────────
            parts = []
            for fpath in top_files:
                abs_path = os.path.join(work_dir, fpath)
                if not os.path.isfile(abs_path):
                    continue
                try:
                    with open(abs_path, encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    # Truncate if too long
                    if len(lines) > max_lines_per_file:
                        lines = lines[:max_lines_per_file]
                        lines.append(f"\n... (truncated, {len(lines)} / total lines shown)\n")
                    parts.append(f"### {fpath}\n```python\n{''.join(lines)}```")
                except Exception:
                    continue

            return "\n\n".join(parts) if parts else "(Could not read files)"

        except subprocess.TimeoutExpired:
            return "(Code retrieval timed out)"
        except Exception as e:
            return f"(Code retrieval failed: {e})"
        finally:
            if work_dir and os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)