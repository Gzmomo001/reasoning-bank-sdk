"""Tests for multi-trajectory scaling induction."""

from reasoning_bank.core.scaling import induce_scaling, _parse_memory_items
from reasoning_bank.llm.base import LLMClient


class _StubLLM(LLMClient):
    def __init__(self, response: str):
        self._response = response

    def chat(self, messages, system=None):
        return self._response


def test_scaling_basic():
    llm = _StubLLM("# Memory Item 1\n## Title Pattern\n## Content Use the successful pattern")
    items = induce_scaling(
        llm,
        "t1",
        "find item",
        [
            {"trajectory": "success traj...", "status": "success"},
            {"trajectory": "fail traj...", "status": "fail"},
        ],
        "web",
    )
    assert len(items) == 1
    assert items[0].status == "mixed"


def test_scaling_multiple_items():
    llm = _StubLLM(
        "# Memory Item 1\n## Title A\n## Content a\n\n"
        "# Memory Item 2\n## Title B\n## Content b"
    )
    items = induce_scaling(
        llm,
        "t2",
        "search",
        [{"trajectory": "traj", "status": "success"}],
        "coding",
    )
    assert len(items) == 2
    assert items[0].domain == "coding"


def test_parse_memory_items_headers():
    raw = "# Memory Item 1\n## Title X\n## Content x\n# Memory Item 2\n## Title Y\n## Content y"
    items = _parse_memory_items(raw)
    assert len(items) == 2
