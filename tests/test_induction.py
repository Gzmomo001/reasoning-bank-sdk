"""Tests for single-trajectory memory induction — real LLM."""

from reasoning_bank.core.induction import _parse_memory_items, induce


async def test_induce_success(llm_with_retry):
    items = await induce(llm_with_retry, "t1", "go home", "think...\naction...", "success", "web")
    assert len(items) >= 1
    assert items[0].status == "success"
    assert items[0].domain == "web"


async def test_induce_fail(llm_with_retry):
    items = await induce(llm_with_retry, "t2", "go home", "think...\naction...", "fail", "web")
    assert len(items) >= 1
    assert items[0].status == "fail"


async def test_induce_coding_domain(llm_with_retry):
    items = await induce(llm_with_retry, "t3", "fix bug", "think...", "success", "coding")
    assert len(items) >= 1
    assert items[0].domain == "coding"


def test_parse_multiple_items():
    raw = "# Memory Item 1\n## Title A\n## Content a\n\n# Memory Item 2\n## Title B\n## Content b"
    items = _parse_memory_items(raw)
    assert len(items) == 2


def test_parse_no_headers():
    raw = "First item\n\nSecond item\n\nThird item"
    items = _parse_memory_items(raw)
    assert len(items) == 3
