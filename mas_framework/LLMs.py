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
import re

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
        self.max_tokens = config.get("max_tokens", 4096)
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
# OpenAI
# ──────────────────────────────────────────────
class OpenAI(LLM):
    _provider_name = "OpenAI"

    def _init_client(self):
        from openai import OpenAI as _OpenAI
        self._client = _OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url or None,
        )

    def call(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=float(self.extra.get("timeout", LLM_TIMEOUT)),
        )

        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)

        duration = time.time() - start
        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ))
        return content


# ──────────────────────────────────────────────
# DeepSeek (OpenAI-compatible, streaming only)
# ──────────────────────────────────────────────
class DeepSeek(LLM):
    _provider_name = "DeepSeek"

    def _init_client(self):
        from openai import OpenAI as _OpenAI
        self._client = _OpenAI(
            base_url=self.base_url,
        )

    def call(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        collected = []
        usage = None

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            timeout=float(self.extra.get("timeout", 180)),
        )

        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta

            piece = None
            if hasattr(delta, "content"):
                piece = delta.content
            elif isinstance(delta, dict):
                piece = delta.get("content")
            elif hasattr(delta, "text"):
                piece = delta.text

            if piece:
                collected.append(piece)

        duration = time.time() - start
        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ))
        return "".join(collected)

# ──────────────────────────────────────────────
# Llama (Ollama / vLLM / any OpenAI-compatible)
# ──────────────────────────────────────────────
class Llama(OpenAI):
    _provider_name = "Llama"


# ──────────────────────────────────────────────
# QWen (Alibaba Cloud, OpenAI-compatible)
# ──────────────────────────────────────────────
class QWen(OpenAI):
    _provider_name = "QWen"


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
# HuggingFace (local inference via transformers)
# ──────────────────────────────────────────────
class HuggingFaceLLM(LLM):
    """
    Runs inference locally using a HuggingFace model loaded via transformers.
    The model and tokenizer are loaded once per hf_model_id and shared across
    all instances (class-level cache), so multiple agents reuse the same weights.
    Config keys:
      hf_model_id    - HuggingFace model ID (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
                       Falls back to `model` if not set.
      max_tokens - max tokens to generate (default 2048)
    """

    _model_cache: dict = {}  # hf_model_id -> {"model": ..., "tokenizer": ...}

    def _init_client(self):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        transformers.logging.set_verbosity_error()

        hf_model_id = self.extra.get("hf_model_id", self.model)

        if hf_model_id not in HuggingFaceLLM._model_cache:
            print(f"[HuggingFaceLLM] Loading model: {hf_model_id}")
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},        # 强制整模型到 GPU0
                attn_implementation="sdpa",
            )
            print(f"[HuggingFaceLLM] Model loaded on device: {model.device}")
            HuggingFaceLLM._model_cache[hf_model_id] = {"model": model, "tokenizer": tokenizer}
        else:
            print(f"[HuggingFaceLLM] Reusing cached model: {hf_model_id}")

        self._client = HuggingFaceLLM._model_cache[hf_model_id]

    def call(self, prompt: str, system_prompt: str = "") -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokenizer = self._client["tokenizer"]
        model = self._client["model"]

        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

        input_len = input_ids.shape[-1]
        max_tokens = int(self.extra.get("max_tokens", 2048))
        do_sample = self.temperature > 0
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=max_tokens,
                temperature=self.temperature if do_sample else None,
                do_sample=do_sample,
            )
        duration = time.time() - start

        new_tokens = output_ids[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 剥离 <think>...</think> 块
        if '<think>' in response:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=input_len,
            output_tokens=len(new_tokens),
        ))
        return response

        self.call_statistics.append(CallStatistic(
            duration=duration,
            input_tokens=input_len,
            output_tokens=len(new_tokens),
        ))
        return response


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────
PROVIDER_MAP = {
    "openai": OpenAI,
    "claude": Claude,
    "deepseek": DeepSeek,
    "ollama": Llama,
    "qwen": QWen,
    "huggingface": HuggingFaceLLM,
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
