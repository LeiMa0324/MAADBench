"""
MAS: Agent + Orchestrator (AutoGen / LangGraph)
- Agent: unified class, uses LLMProvider
- Orchestrator: AutoGen or LangGraph backend
- Architecture: sequential, parallel, hierarchical, centered
- Same interface regardless of backend or architecture
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple

from dataclasses import dataclass
from abc import ABC, abstractmethod
import yaml
from mas_framework.LLMs import *
from  datetime import datetime
import logging



# ══════════════════════════════════════════════
# Agent Output
# ══════════════════════════════════════════════
@dataclass
class AgentOutput:
    agent_role: str
    content: Dict[str, Any]
    confidence: float = 0.7
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "agent_role": self.agent_role,
            "content": self.content,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentOutput":
        return cls(
            agent_role=d["agent_role"],
            content=d["content"],
            confidence=d.get("confidence", 0.7),
            reasoning=d.get("reasoning", ""),
        )


# ══════════════════════════════════════════════
# Unified Agent
# ══════════════════════════════════════════════
class Agent:
    """
    Single agent with a role and an LLM.
    Same class used across all backends and architectures.
    """

    def __init__(
        self,
        role: str,
        llm_config: dict,
        role_prompt: Optional[str] = None,
    ):
        self.role = role
        self.role_prompt = role_prompt or f"You are a {role}."
        self._llm: LLM = create_llm_from_config(llm_config)
        self.history: List[AgentOutput] = []

    @property
    def last_call_statistic(self) -> CallStatistic:
        return self._llm.call_statistics[-1]

    def call_llm(self, user_prompt: str) -> str:
        return self._llm.call(prompt=user_prompt, system_prompt=self.role_prompt)

    def execute(self, input_data: Dict[str, Any]) -> AgentOutput:
        problem = input_data.get("problem", "")
        context = input_data.get("context", "")
        previous_outputs = input_data.get("previous_outputs", {})
        prior = {k: v.to_dict() for k, v in previous_outputs.items()}

        user_prompt = (
            f"Problem: {problem}\n\n"
            f"Context: {context}\n\n"
            f"Previous outputs:\n{json.dumps(prior, indent=2)}\n\n"
            "Respond with a JSON object. Include a 'confidence' field between 0 and 1 if possible."
        )

        response = self.call_llm(user_prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> AgentOutput:

        def _extract_json_text(resp: str) -> str:
            # 1. Explicit ```json block
            if "```json" in resp:
                return resp.split("```json", 1)[1].split("```", 1)[0].strip()

            # 2. Generic ``` block, but only if content looks like JSON
            if "```" in resp:
                inner = resp.split("```", 1)[1].split("```", 1)[0].strip()
                if inner.startswith("{") or inner.startswith("["):
                    return inner

            # 3. Find outermost { ... } via brace matching
            start = resp.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(resp)):
                    if resp[i] == "{":
                        depth += 1
                    elif resp[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return resp[start:i + 1]

            return resp.strip()

        def _coerce_to_obj(s: str):
            """
            Try to coerce s into a Python object (dict/list/primitive),
            handling:
            1) normal JSON
            2) double-encoded JSON string
            3) escaped JSON without outer quotes: {\\\"a\\\":1}
            """
            s = s.strip()

            # Try 1: normal JSON
            try:
                obj = json.loads(s)
            except Exception:
                obj = None

            # Try 2: double-encoded JSON string
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    pass

            if obj is not None:
                return obj

            # Try 3: escaped-json-without-quotes, e.g. {\"a\":1}
            # Heuristic: contains \", and starts with {\" or [\"
            if '\\"' in s and (
                    s.startswith('{\\\"') or s.startswith('[\\\"') or s.startswith('\\{\"') or s.startswith('\\[')):
                # Some models prepend a backslash before { ... } ; normalize
                if s.startswith('\\'):
                    s = s[1:]

                # Wrap as a JSON string literal to unescape safely
                try:
                    unescaped = json.loads(f"\"{s}\"")  # turns {\"a\":1} -> {"a":1}
                    obj2 = json.loads(unescaped)
                    return obj2
                except Exception:
                    pass

            # Last resort: sometimes it's almost JSON but with leading/trailing junk
            # (you can add regex extraction later if needed)
            raise ValueError("Unable to coerce model output into JSON")

        try:
            json_str = _extract_json_text(response)
            parsed = _coerce_to_obj(json_str)

            if isinstance(parsed, dict):
                confidence = parsed.pop("confidence", 0.7)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                content = parsed
            else:
                confidence = 0.7
                content = {"result": parsed}

            output = AgentOutput(
                agent_role=self.role,
                content=content,
                confidence=confidence,
                reasoning=response,
            )

        except Exception as e:
            output = AgentOutput(
                agent_role=self.role,
                content={"error": "Failed to parse", "raw": response, "exception": str(e)},
                confidence=0.2,
                reasoning=response,
            )

        self.history.append(output)
        return output


    def build_input(
        self,
        problem: str,
        previous_outputs: Dict[str, AgentOutput],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "problem": problem if context is None else f"{problem}\n\nContext: {context}",
            "previous_outputs": previous_outputs,
        }

    def __repr__(self):
        return f"Agent(role={self.role}, llm={self._llm})"


# ══════════════════════════════════════════════
# Abstract Orchestrator
# ══════════════════════════════════════════════

T = TypeVar("T", bound="MASOrchestrator")
class MASOrchestrator(ABC):
    """
    Abstract MAS orchestrator.
    All backends and architectures expose the same run() interface.
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        architecture: str,
        coordinator_role: Optional[str] = None,
    ):
        self.agents = agents
        self.architecture = architecture
        self.coordinator_role = coordinator_role  # role name of supervisor/hub agent
        self.trace: List[Dict] = []


    def get_coordinator_agent(self) -> Agent:
        return self.agents[self.coordinator_role]

    def get_worker_agents(self) -> List[Agent]:
        return [a for name, a in self.agents.items() if name != self.coordinator_role]

    def save_execution_trace(self,meta_data:dict, output_dir: str = "traces", trace_id: str = None) -> Tuple[Dict, str]:
        os.makedirs(output_dir, exist_ok=True)

        if trace_id is None:
            trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 自动合并 self.trace
        trace_payload = {
            **meta_data,  # config, question, etc.
            "trace": self.trace  # 强制使用 self.trace
        }
        path = os.path.join(output_dir, f"{trace_id}.json")

        with open(path, "w") as f:
            json.dump(trace_payload, f, indent=2)

        print(f"Execution trace saved to: {path}")
        return trace_payload

    def run(self, problem, context: Optional[str] = None):
        self.trace = []

        if self.architecture == "sequential":
            outputs = self._run_sequential(problem, context)
        elif self.architecture == "parallel":
            outputs = self._run_parallel(problem, context)
        elif self.architecture == "hierarchical":
            outputs = self._run_hierarchical(problem, context)
        elif self.architecture == "centralized":
            outputs = self._run_centered(problem, context)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")


        last_trace = self.trace[-1] if self.trace else None
        final_answer = None
        if last_trace:
            output_dict = last_trace["output"]  # 这是 .to_dict() 的结果
            final_answer = output_dict.get("content", {}).get("final_answer")

        return {
            "problem": problem,
            "final_answer": final_answer,
            "success": final_answer is not None,
            "architecture": self.architecture,
            "trace": self.trace,
            "agent_confidences": {k: v.confidence for k, v in outputs.items() if isinstance(v, AgentOutput)},
        }

    def _run_sequential(self, problem, context: Optional[str] = None):
        pass

    def _run_hierarchical(self, problem, context: Optional[str] = None):
        pass

    def _run_centered(self, problem, context: Optional[str] = None):
        pass

    @classmethod
    def from_config(cls: Type[T], config_path: str) -> "T":
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        system_config = config.get("system", {})
        agents_config = config.get("agents", [])
        llm_config = config.get("llm", {})

        agents: Dict[str, Agent] = {}

        for agent_cfg in agents_config:
            agent_llm_config = {**llm_config}
            if "max_tokens" in agent_cfg:
                agent_llm_config["max_tokens"] = agent_cfg["max_tokens"]
            agent = Agent(
                role=agent_cfg["role"],
                role_prompt=agent_cfg.get("prompt"),
                llm_config=agent_llm_config,
            )
            agents[agent_cfg["role"]] = agent

        coordinator_role = system_config.get("coordinator")

        return cls(
            agents=agents,
            architecture=system_config.get("architecture", "sequential"),
            coordinator_role=coordinator_role,
        )


# ══════════════════════════════════════════════
# AutoGen Backend
# ══════════════════════════════════════════════
class AutoGenMAS(MASOrchestrator):
    """
    AutoGen-backed MAS.
    Uses ConversableAgent + GroupChat for multi-agent orchestration.
    Our Agent handles all LLM calls; AutoGen handles message routing.
    """

    def __init__(self, agents, architecture, coordinator_role=None):
        super().__init__(agents, architecture, coordinator_role)
        self._setup_autogen()

    def _setup_autogen(self):
        try:
            from autogen import ConversableAgent, GroupChat, GroupChatManager

            self._autogen_imports = {
                "ConversableAgent": ConversableAgent,
                "GroupChat": GroupChat,
                "GroupChatManager": GroupChatManager,
            }
        except ImportError:
            raise ImportError("pip install pyautogen")

        # Create AutoGen agents that delegate LLM calls to our Agent
        self._autogen_agents = {}
        ConversableAgent = self._autogen_imports["ConversableAgent"]

        for name, agent in self.agents.items():
            def make_reply_func(ag):
                def reply_func(recipient, messages, sender, config):
                    last_msg = messages[-1]["content"] if messages else ""
                    response = ag.call_llm(last_msg)
                    return True, response
                return reply_func

            autogen_agent = ConversableAgent(
                name=name,
                system_message=agent.role_prompt,
                llm_config=False,  # disable AutoGen's built-in LLM
                human_input_mode="NEVER",
            )
            autogen_agent.register_reply([ConversableAgent, None], make_reply_func(agent))
            self._autogen_agents[name] = autogen_agent



    # ── Sequential ─────────────────────────────
    def _run_sequential(self, problem, context: Optional[str] = None):
        """A → B → C using AutoGen two-agent chat chaining."""
        outputs = {}
        agent_list = list(self.agents.values())
        for i, agent in enumerate(agent_list):
            # Build prompt with all previous outputs
            input_data = agent.build_input(problem, outputs, context)
            output = agent.execute(input_data)
            outputs[agent.role] = output
            self.trace.append({
                "agent": agent.role,
                "phase": "sequential",
                "output": output.to_dict(),
                "call_statistic": agent.last_call_statistic.to_dict(),
            })

            # Removed unnecessary AutoGen chat between consecutive agents

        return outputs


    # ── Hierarchical ───────────────────────────
    def _run_hierarchical(self, problem, context: Optional[str] = None):
        """Supervisor plans → workers execute → supervisor summarizes."""
        outputs = {}
        supervisor = self.get_coordinator_agent()
        workers = self.get_worker_agents()

        # Phase 1: supervisor plans
        plan_input = supervisor.build_input(problem, {}, context)
        plan_output = supervisor.execute(plan_input)
        outputs[f"{supervisor.role}_plan"] = plan_output
        self.trace.append({
            "agent": supervisor.role,
            "phase": "planning",
            "output": plan_output.to_dict(),
            "call_statistic": supervisor.last_call_statistic.to_dict(),
        })

        # Phase 2: workers execute (with plan as context)
        for agent in workers:
            input_data = agent.build_input(problem, outputs, context)
            output = agent.execute(input_data)
            outputs[agent.role] = output
            self.trace.append({
                "agent": agent.role,
                "phase": "execution",
                "output": output.to_dict(),
                "call_statistic": agent.last_call_statistic.to_dict(),
            })

            # Removed unnecessary AutoGen supervisor-to-worker chat

        # Phase 3: supervisor summarizes
        summary_input = supervisor.build_input(
            f"Summarize and synthesize results for: {problem}",
            outputs,
            context,
        )
        summary_output = supervisor.execute(summary_input)
        outputs[f"{supervisor.role}_summary"] = summary_output
        self.trace.append({
            "agent": supervisor.role,
            "phase": "summarization",
            "output": summary_output.to_dict(),
            "call_statistic": supervisor.last_call_statistic.to_dict(),
        })

        return outputs

    # ── Centered (Star) ────────────────────────
    def _run_centered(self, problem, context: Optional[str] = None):
        """Hub communicates with each worker individually, then synthesizes."""
        outputs = {}
        hub = self.get_coordinator_agent()
        workers = self.get_worker_agents()

        # Phase 1: hub consults each worker one-on-one
        for agent in workers:
            # Hub asks worker
            input_data = agent.build_input(problem, {}, context)
            output = agent.execute(input_data)
            outputs[agent.role] = output
            self.trace.append({
                "agent": agent.role,
                "phase": "consultation",
                "output": output.to_dict(),
                "call_statistic": agent.last_call_statistic.to_dict(),
            })

            # Removed unnecessary AutoGen hub-worker chat

        # Phase 2: hub synthesizes all worker outputs
        hub_input = hub.build_input(problem, outputs, context)
        hub_output = hub.execute(hub_input)
        outputs[hub.role] = hub_output
        self.trace.append({
            "agent": hub.role,
            "phase": "synthesis",
            "output": hub_output.to_dict(),
            "call_statistic": hub.last_call_statistic.to_dict(),
        })

        return outputs


# ══════════════════════════════════════════════
# LangGraph Backend
# ══════════════════════════════════════════════
class LangGraphMAS(MASOrchestrator):
    """
    LangGraph-backed MAS.
    Uses StateGraph for workflow orchestration.
    Our Agent handles all LLM calls; LangGraph handles state and routing.
    """

    def __init__(self, agents, architecture, coordinator_role=None):
        super().__init__(agents, architecture, coordinator_role)
        try:
            from langgraph.graph import StateGraph, END
            self._StateGraph = StateGraph
            self._END = END
        except ImportError:
            raise ImportError("pip install langgraph")

    def _make_node(self, agent: Agent, phase: str, input_builder):
        """Create a LangGraph node function for an agent."""
        def node(state: dict) -> dict:
            outputs = state.get("outputs", {})

            # Reconstruct AgentOutput from state
            prev = {}
            for k, v in outputs.items():
                if isinstance(v, AgentOutput):
                    prev[k] = v
                elif isinstance(v, dict) and "agent_role" in v:
                    prev[k] = AgentOutput.from_dict(v)

            input_data = input_builder(state["problem"], prev, state.get("context"))
            output = agent.execute(input_data)

            new_outputs = dict(outputs)
            new_outputs[agent.role] = output

            self.trace.append({
                "agent": agent.role,
                "phase": phase,
                "output": output.to_dict(),
                "call_statistic": agent.last_call_statistic.to_dict(),
            })

            return {**state, "outputs": new_outputs}
        return node

    # ── Sequential ─────────────────────────────
    def _run_sequential(self, problem, context: Optional[str] = None):
        """A → B → C as a linear StateGraph."""
        from typing import TypedDict

        class State(TypedDict):
            problem: str
            context: Optional[str]
            outputs: Dict[str, Any]

        graph = self._StateGraph(State)
        agent_names = list(self.agents.keys())

        # Add nodes
        for name in agent_names:
            agent = self.agents[name]
            node = self._make_node(agent, "sequential", agent.build_input)
            graph.add_node(name, node)

        # Wire edges: linear chain
        graph.set_entry_point(agent_names[0])
        for i in range(len(agent_names) - 1):
            graph.add_edge(agent_names[i], agent_names[i + 1])
        graph.add_edge(agent_names[-1], self._END)

        compiled = graph.compile()
        result = compiled.invoke({
            "problem": problem,
            "context": context,
            "outputs": {},
        })

        return self._extract_outputs(result)

    # ── Hierarchical ───────────────────────────
    def _run_hierarchical(self, problem, context: Optional[str] = None):
        """Supervisor_plan → workers → supervisor_summary."""
        from typing import TypedDict

        class State(TypedDict):
            problem: str
            context: Optional[str]
            outputs: Dict[str, Any]

        graph = self._StateGraph(State)
        supervisor = self.get_coordinator_agent()
        workers = self.get_worker_agents()

        # Node: supervisor plans
        plan_name = f"{supervisor.role}_plan"

        def plan_node(state: dict) -> dict:
            input_data = supervisor.build_input(state["problem"], {}, state.get("context"))
            output = supervisor.execute(input_data)
            new_outputs = dict(state.get("outputs", {}))
            new_outputs[plan_name] = output
            self.trace.append({
                "agent": supervisor.role,
                "phase": "planning",
                "output": output.to_dict(),
                "call_statistic": supervisor.last_call_statistic.to_dict(),
            })
            return {**state, "outputs": new_outputs}

        graph.add_node(plan_name, plan_node)

        # Nodes: workers execute
        for w in workers:
            node = self._make_node(w, "execution", w.build_input)
            graph.add_node(w.role, node)

        # Node: supervisor summarizes
        summary_name = f"{supervisor.role}_summary"

        def summary_node(state: dict) -> dict:
            outputs = state.get("outputs", {})
            prev = {}
            for k, v in outputs.items():
                if isinstance(v, AgentOutput):
                    prev[k] = v
                elif isinstance(v, dict) and "agent_role" in v:
                    prev[k] = AgentOutput.from_dict(v)

            input_data = supervisor.build_input(
                f"Summarize and synthesize results for: {state['problem']}",
                prev,
                state.get("context"),
            )
            output = supervisor.execute(input_data)
            new_outputs = dict(outputs)
            new_outputs[summary_name] = output
            self.trace.append({
                "agent": supervisor.role,
                "phase": "summarization",
                "output": output.to_dict(),
                "call_statistic": supervisor.last_call_statistic.to_dict(),
            })
            return {**state, "outputs": new_outputs}

        graph.add_node(summary_name, summary_node)

        # Wire: plan → workers (sequential) → summary
        graph.set_entry_point(plan_name)
        prev_node = plan_name
        for w in workers:
            graph.add_edge(prev_node, w.role)
            prev_node = w.role
        graph.add_edge(prev_node, summary_name)
        graph.add_edge(summary_name, self._END)

        compiled = graph.compile()
        result = compiled.invoke({
            "problem": problem,
            "context": context,
            "outputs": {},
        })

        return self._extract_outputs(result)

    # ── Centered (Star) ────────────────────────
    def _run_centered(self, problem, context: Optional[str] = None):
        """Hub ↔ Worker1 → Hub ↔ Worker2 → ... → Hub synthesizes."""
        from typing import TypedDict

        class State(TypedDict):
            problem: str
            context: Optional[str]
            outputs: Dict[str, Any]

        graph = self._StateGraph(State)
        hub = self.get_coordinator_agent()
        workers = self.get_worker_agents()

        # For each worker: hub_ask → worker_reply pair
        node_sequence = []
        for w in workers:
            # Worker consultation node (hub asks, worker answers)
            consult_name = f"consult_{w.role}"

            def make_consult_node(worker):
                def node(state: dict) -> dict:
                    # Worker answers independently
                    input_data = worker.build_input(state["problem"], {}, state.get("context"))
                    output = worker.execute(input_data)
                    new_outputs = dict(state.get("outputs", {}))
                    new_outputs[worker.role] = output
                    self.trace.append({
                        "agent": worker.role,
                        "phase": "consultation",
                        "output": output.to_dict(),
                        "call_statistic": worker.last_call_statistic.to_dict(),
                    })
                    return {**state, "outputs": new_outputs}
                return node

            graph.add_node(consult_name, make_consult_node(w))
            node_sequence.append(consult_name)

        # Hub synthesis node
        synthesis_name = f"{hub.role}_synthesis"

        def synthesis_node(state: dict) -> dict:
            outputs = state.get("outputs", {})
            prev = {}
            for k, v in outputs.items():
                if isinstance(v, AgentOutput):
                    prev[k] = v
                elif isinstance(v, dict) and "agent_role" in v:
                    prev[k] = AgentOutput.from_dict(v)

            input_data = hub.build_input(state["problem"], prev, state.get("context"))
            output = hub.execute(input_data)
            new_outputs = dict(outputs)
            new_outputs[hub.role] = output
            self.trace.append({
                "agent": hub.role,
                "phase": "synthesis",
                "output": output.to_dict(),
                "call_statistic": hub.last_call_statistic.to_dict(),
            })
            return {**state, "outputs": new_outputs}

        graph.add_node(synthesis_name, synthesis_node)
        node_sequence.append(synthesis_name)

        # Wire: linear chain of consultations → synthesis
        graph.set_entry_point(node_sequence[0])
        for i in range(len(node_sequence) - 1):
            graph.add_edge(node_sequence[i], node_sequence[i + 1])
        graph.add_edge(node_sequence[-1], self._END)

        compiled = graph.compile()
        result = compiled.invoke({
            "problem": problem,
            "context": context,
            "outputs": {},
        })

        return self._extract_outputs(result)

    # ── Helper ─────────────────────────────────
    def _extract_outputs(self, result: dict) -> Dict[str, AgentOutput]:
        outputs = {}
        for k, v in result.get("outputs", {}).items():
            if isinstance(v, AgentOutput):
                outputs[k] = v
            elif isinstance(v, dict) and "agent_role" in v:
                outputs[k] = AgentOutput.from_dict(v)
        return outputs



# ══════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════
BACKEND_MAP = {
    "autogen": AutoGenMAS,
    "langgraph": LangGraphMAS,
}


# ══════════════════════════════════════════════
# Quick test (no API calls, just structure)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    pass

