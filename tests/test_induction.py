"""Tests for single-trajectory memory induction."""

from reasoning_bank.core.induction import induce, _parse_memory_items
from reasoning_bank.llm.base import LLMClient


class _StubLLM(LLMClient):
    def __init__(self, response: str):
        self._response = response

    def chat(self, messages, system=None):
        return self._response


def test_induce_success():
    llm = _StubLLM("# Memory Item 1\n## Title Click nav\n## Content Click the nav element")
    items = induce(llm, "t1", "go home", "think...\naction...", "success", "web")
    assert len(items) == 1
    assert items[0].status == "success"
    assert items[0].domain == "web"


def test_induce_fail():
    llm = _StubLLM("# Memory Item 1\n## Title Avoid X\n## Content Don't do X")
    items = induce(llm, "t2", "go home", "think...\naction...", "fail", "web")
    assert len(items) == 1
    assert items[0].status == "fail"


def test_induce_coding_domain():
    llm = _StubLLM("# Memory Item 1\n## Title Read tests\n## Content Always run tests first")
    items = induce(llm, "t3", "fix bug", "think...", "success", "coding")
    assert items[0].domain == "coding"


def test_parse_multiple_items():
    raw = "# Memory Item 1\n## Title A\n## Content a\n\n# Memory Item 2\n## Title B\n## Content b"
    items = _parse_memory_items(raw)
    assert len(items) == 2


def test_parse_no_headers():
    raw = "First item\n\nSecond item\n\nThird item"
    items = _parse_memory_items(raw)
    assert len(items) == 3
