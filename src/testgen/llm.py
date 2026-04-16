"""Minimal LLM abstraction used by testgen modules.

Any object with ``.invoke(prompt: str) -> str`` is a valid llm. ``get_llm()`` returns
a LangChain chat model if available, otherwise a deterministic ``MockLLM`` useful for
tests and dry runs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol


class LLMLike(Protocol):
    def invoke(self, prompt: str) -> str: ...


class MockLLM:
    """Deterministic offline LLM. Returns structured stub text keyed by prompt hash."""

    def __init__(self, tag: str = "mock") -> None:
        self.tag = tag
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        h = hashlib.md5(prompt.encode()).hexdigest()[:6]
        p_lower = prompt.lower()
        if "entities" in p_lower and "facts" in p_lower:
            return (
                "ENTITIES:\n- entity_a\n- entity_b\n"
                "FACTS:\n- entity_a is related to entity_b.\n"
                "RELATIONS:\n- entity_a | relates_to | entity_b | evidence sentence."
            )
        if "generate a" in p_lower and "question" in p_lower:
            return f"What is the relationship between the items described (case {h})?"
        if "answer the question" in p_lower:
            return "A concise grounded answer."
        if "grounded" in p_lower or "supported" in p_lower:
            return "GROUNDED: YES\nCONFIDENCE: 0.9"
        if "evolve" in p_lower or "reasoning" in p_lower or "conditional" in p_lower:
            return f"Evolved variant of the question ({h})."
        return f"[{self.tag}:{h}] " + prompt[:80]


class _LCWrapper:
    """Wraps a LangChain BaseChatModel into an object with ``.invoke(str) -> str``."""

    def __init__(self, model: Any) -> None:
        self._m = model

    def invoke(self, prompt: str) -> str:
        try:
            from langchain_core.messages import HumanMessage  # type: ignore
            msg = self._m.invoke([HumanMessage(content=prompt)])
        except Exception:
            msg = self._m.invoke(prompt)
        content = getattr(msg, "content", msg)
        if isinstance(content, list):
            parts = [getattr(p, "text", str(p)) for p in content]
            return "".join(parts)
        return str(content)


def get_llm(model: str | None = None, provider: str | None = None) -> LLMLike:
    """Return a real LLM wrapper if credentials/deps are available, else MockLLM.

    Provider resolution order (first match wins):
      1. Explicit `provider` argument.
      2. `LLM_PROVIDER` env var. Values: local | ollama | anthropic | openai.
      3. `local` when `OLLAMA_HOST` or `LLM_BASE_URL` is set.
      4. `anthropic` when `ANTHROPIC_API_KEY` is set.
      5. `openai` when `OPENAI_API_KEY` is set.
      6. Auto-probe `http://localhost:11434` — fall back to local Ollama if up.
      7. MockLLM.

    Local path uses the OpenAI-compatible REST surface (`/v1/chat/completions`)
    so the same wiring works with Ollama, LM Studio, vLLM, and llama.cpp
    server. Override with env:
      - LLM_BASE_URL   (default: http://localhost:11434/v1  — Ollama)
      - LLM_MODEL      (default: qwen2.5:7b-instruct)
      - LLM_API_KEY    (default: "ollama", most local servers ignore it)
      - LLM_TEMPERATURE
    """
    import os

    prov = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not prov:
        if os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL"):
            prov = "local"
        elif os.getenv("ANTHROPIC_API_KEY"):
            prov = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            prov = "openai"
        else:
            prov = "local" if _probe_local_llm() else "mock"

    try:
        if prov in ("local", "ollama", "lmstudio", "vllm", "llamacpp"):
            return _build_local_llm(model)
        if prov == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return _LCWrapper(
                ChatAnthropic(model=model or "claude-sonnet-4-5", temperature=0)
            )
        if prov == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI  # type: ignore
            return _LCWrapper(ChatOpenAI(model=model or "gpt-4o-mini", temperature=0))
    except Exception as exc:  # pragma: no cover — surfaced via MockLLM fallback
        import logging

        logging.getLogger(__name__).warning("LLM init failed (%s); falling back: %s", prov, exc)
    return MockLLM()


def _probe_local_llm(timeout: float = 0.4) -> bool:
    """Quick TCP probe — returns True iff Ollama (or a compatible server) answers."""
    import os
    from urllib.parse import urlparse
    import urllib.request

    base = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    # Ollama health is at /api/tags, OpenAI-compatible servers expose /v1/models.
    # Try both: whichever responds first is good enough.
    host = urlparse(base).netloc or "localhost:11434"
    for probe in (f"http://{host}/api/tags", f"{base.rstrip('/')}/models"):
        try:
            req = urllib.request.Request(probe, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - local
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            continue
    return False


def _build_local_llm(model_override: str | None) -> LLMLike:
    """Wire an OpenAI-compatible local endpoint (default: Ollama)."""
    import os

    from langchain_openai import ChatOpenAI  # type: ignore

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = model_override or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    chat = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        timeout=120,
        max_retries=1,
    )
    return _LCWrapper(chat)


def parse_json_block(text: str) -> Any:
    """Best-effort JSON extraction from LLM output. Returns None on failure."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None
