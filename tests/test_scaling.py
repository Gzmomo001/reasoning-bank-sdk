"""Tests for multi-trajectory scaling induction — real LLM."""

from reasoning_bank.core.scaling import induce_scaling, _parse_memory_items


def test_scaling_basic(llm_with_retry):
    items = induce_scaling(
        llm_with_retry, "t1", "find item",
        [
            {"trajectory": "success traj...", "status": "success"},
            {"trajectory": "fail traj...", "status": "fail"},
        ],
        "web",
    )
    assert len(items) >= 1
    assert items[0].status == "mixed"


def test_scaling_multiple_items(llm_with_retry):
    items = induce_scaling(
        llm_with_retry, "t2", "search",
        [{"trajectory": "traj", "status": "success"}],
        "coding",
    )
    assert len(items) >= 1
    assert items[0].domain == "coding"


def test_parse_memory_items_headers():
    raw = "# Memory Item 1\n## Title X\n## Content x\n# Memory Item 2\n## Title Y\n## Content y"
    items = _parse_memory_items(raw)
    assert len(items) == 2
