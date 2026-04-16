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


def get_llm(model: str | None = None, provider: str = "anthropic") -> LLMLike:
    """Return a real LLM wrapper if credentials/deps are available, else MockLLM."""
    import os

    try:
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return _LCWrapper(ChatAnthropic(model=model or "claude-sonnet-4-5", temperature=0))
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI  # type: ignore
            return _LCWrapper(ChatOpenAI(model=model or "gpt-4o-mini", temperature=0))
    except Exception:
        pass
    return MockLLM()


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
