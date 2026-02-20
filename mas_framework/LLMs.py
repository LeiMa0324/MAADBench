"""
LLM Provider Abstraction Layer (Simplified)
One config dict in → one usable LLM instance out.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

LLM_TIMEOUT = 120  # seconds


@dataclass
class CallStatistic:
    duration: float
    input_tokens: int
    output_tokens: int
    timed_out: bool = False
    timeout_in_reasoning: bool = False  # True if timeout occurred while model was still in reasoning chain

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timed_out": self.timed_out,
            "superlong_reasoning": self.timeout_in_reasoning,
        }


# ──────────────────────────────────────────────
# Abstract Base Class
# ──────────────────────────────────────────────
class LLM(ABC):
    """
    Abstract base class for all LLM providers.
    Subclasses only need to implement _init_client() and call().
    """

    def __init__(self, config: dict):
        self.model = config["model"]
        self.base_url = config.get("base_url", "")
        self.temperature = config.get("temperature", 0.0)
        # self.max_tokens = config.get("max_tokens", 2048)
        self.extra = {
            k: v for k, v in config.items()
            if k not in {"provider", "model", "base_url", "temperature"}
        }
        self._client = None
        self._init_client()
        self.call_statistics: list[CallStatistic] = []

    @abstractmethod
    def _init_client(self):
        pass

    @abstractmethod
    def call(self, prompt: str, system_prompt: str = "") -> str:
        pass

    def _handle_timeout(self, provider: str, duration: float, in_reasoning: bool,
                        partial_reasoning: str = "") -> str:
        msg = f"[{provider}] call timed out after {duration:.1f}s"
        if in_reasoning:
            msg += f" (in reasoning chain, {len(partial_reasoning)} chars collected)"
            logger.warning(msg)
            if partial_reasoning:
                logger.warning(f"[{provider}] partial reasoning:\n{partial_reasoning}")
        else:
            logger.warning(msg)
        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=0,
            output_tokens=0,
            timed_out=True,
            timeout_in_reasoning=in_reasoning,
        ))
        import json
        result = {"error": "LLM call timed out"}
        if partial_reasoning:
            result["partial_reasoning"] = partial_reasoning
        return json.dumps(result)

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model})"


# ──────────────────────────────────────────────
# OpenAI-compatible Base (OpenAI / DeepSeek / Llama / QWen)
# ──────────────────────────────────────────────
class OpenAICompatibleLLM(LLM):
    """
    Shared call() logic for all OpenAI chat-completions-compatible backends.
    Subclasses only need to implement _init_client() and set _provider_name.
    """
    _provider_name: str = "OpenAI-compatible"

    def _is_timeout_error(self, error: Exception) -> bool:
        name = type(error).__name__.lower()
        message = str(error).lower()
        return "timeout" in name or "timeout" in message or "timed out" in message

    def call(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_timeout = float(self.extra.get("timeout", LLM_TIMEOUT))
        hard_timeout = float(self.extra.get("hard_timeout", max(request_timeout * 3, request_timeout)))
        max_retries = int(self.extra.get("max_retries", 0))

        collected_content = []
        collected_reasoning = []  # populated by reasoning models (e.g. DeepSeek-R1, o-series)
        usage = None
        start = time.time()

        for attempt in range(max_retries + 1):
            collected_content = []
            collected_reasoning = []
            usage = None
            attempt_start = time.time()
            try:
                if self._provider_name=='DeepSeek':
                    stream = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        timeout=request_timeout,
                        stream=True,
                        stream_options={"include_usage": True},
                        max_tokens= 2046  # limit deep think of deepseek
                    )
                else:
                    stream = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        timeout=request_timeout,
                        stream=True,
                        stream_options={"include_usage": True},
                    )

                for chunk in stream:
                    if time.time() - attempt_start > hard_timeout:
                        raise TimeoutError(
                            f"Exceeded hard timeout ({hard_timeout:.1f}s) for one request attempt."
                        )
                    if chunk.usage:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        collected_content.append(delta.content)
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        collected_reasoning.append(delta.reasoning_content)
                break
            except Exception as e:
                if self._is_timeout_error(e):
                    if attempt < max_retries:
                        retry_wait = min(2 ** attempt, 8)
                        logger.warning(
                            "[%s] timeout on attempt %s/%s; retrying in %.1fs",
                            self._provider_name, attempt + 1, max_retries + 1, retry_wait
                        )
                        time.sleep(retry_wait)
                        continue
                    in_reasoning = bool(collected_reasoning) and not bool(collected_content)
                    return self._handle_timeout(
                        self._provider_name, time.time() - start, in_reasoning,
                        partial_reasoning="".join(collected_reasoning),
                    )
                raise

        duration = time.time() - start
        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ))
        return "".join(collected_content)


# ──────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────
class OpenAI(OpenAICompatibleLLM):
    _provider_name = "OpenAI"

    def _init_client(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )


# ──────────────────────────────────────────────
# DeepSeek (OpenAI-compatible)
# ──────────────────────────────────────────────
class DeepSeek(OpenAICompatibleLLM):
    _provider_name = "DeepSeek"

    def _init_client(self):
        import httpx
        from openai import OpenAI

        # DeepSeek latency is often spiky: use longer read timeout + retries by default.
        self.extra.setdefault("timeout", 180)
        self.extra.setdefault("hard_timeout", 600)
        self.extra.setdefault("max_retries", 2)

        self._client = OpenAI(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=30.0,
                read=float(self.extra["timeout"]),
                write=30.0,
                pool=30.0,
            ),
        )

# ──────────────────────────────────────────────
# Llama (Ollama / vLLM / any OpenAI-compatible)
# ──────────────────────────────────────────────
class Llama(OpenAICompatibleLLM):
    _provider_name = "Llama"

    def _init_client(self):
        from openai import OpenAI
        self._client = OpenAI(
            # api_key=self.api_key or "ollama",
            base_url=self.base_url
        )


# ──────────────────────────────────────────────
# QWen (Alibaba Cloud, OpenAI-compatible)
# ──────────────────────────────────────────────
class QWen(OpenAICompatibleLLM):
    _provider_name = "QWen"

    def _init_client(self):
        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.base_url
        )


# ──────────────────────────────────────────────
# Claude (Anthropic)
# ──────────────────────────────────────────────
class Claude(LLM):

    def _init_client(self):
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def call(self, prompt: str, system_prompt: str = "") -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.extra.get("max_tokens", 8192),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        collected_text = []
        collected_thinking = []  # populated when extended thinking is enabled
        start = time.time()
        try:
            with self._client.messages.stream(**kwargs) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_delta":
                        delta_type = getattr(event.delta, "type", None)
                        if delta_type == "thinking_delta":
                            collected_thinking.append(getattr(event.delta, "thinking", ""))
                        elif delta_type == "text_delta":
                            collected_text.append(getattr(event.delta, "text", ""))
                usage = stream.get_final_message().usage
        except Exception as e:
            if "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower():
                in_reasoning = bool(collected_thinking) and not bool(collected_text)
                return self._handle_timeout(
                    "Claude", time.time() - start, in_reasoning,
                    partial_reasoning="".join(collected_thinking),
                )
            raise
        duration = time.time() - start
        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        ))
        return "".join(collected_text)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────
PROVIDER_MAP = {
    "openai": OpenAI,
    "claude": Claude,
    "deepseek": DeepSeek,
    "ollama": Llama,
    "qwen": QWen,
}

def create_llm_from_config(config: dict) -> LLM:
    """
    Create LLM from a single provider config dict that includes the 'provider' key.
    Designed to work directly with YAML config entries.
    """
    provider = config.get("provider")
    if not provider:
        raise ValueError("Config dict must contain a 'provider' key.")

    provider_key = provider.lower()
    if provider_key not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {list(PROVIDER_MAP.keys())}"
        )
    return PROVIDER_MAP[provider_key](config)

# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    pass
