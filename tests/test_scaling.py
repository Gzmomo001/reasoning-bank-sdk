"""Tests for multi-trajectory scaling induction."""

from reasoning_bank.core.parsing import parse_memory_items
from reasoning_bank.core.scaling import induce_scaling


async def test_scaling_basic(llm_with_retry):
    items = await induce_scaling(
        llm_with_retry,
        "t1",
        "find item",
        [
            {"trajectory": "success traj...", "status": "success"},
            {"trajectory": "fail traj...", "status": "fail"},
        ],
        "web",
    )
    assert len(items) >= 1
    assert items[0].status == "mixed"


async def test_scaling_multiple_items(llm_with_retry):
    items = await induce_scaling(
        llm_with_retry,
        "t2",
        "search",
        [{"trajectory": "traj", "status": "success"}],
        "coding",
    )
    assert len(items) >= 1
    assert items[0].domain == "coding"


async def test_scaling_returns_few_items(llm_with_retry):
    """Regression test: scaling should return 1-5 items, not dozens."""
    items = await induce_scaling(
        llm_with_retry,
        "t3",
        "compare",
        [
            {"trajectory": "good path...", "status": "success"},
            {"trajectory": "bad path...", "status": "fail"},
        ],
        "web",
    )
    assert 1 <= len(items) <= 5, f"Expected 1-5 items, got {len(items)}"


def test_parse_memory_items_headers():
    raw = "# Memory Item 1\n## Title X\n## Content x\n# Memory Item 2\n## Title Y\n## Content y"
    items = parse_memory_items(raw)
    assert len(items) == 2
