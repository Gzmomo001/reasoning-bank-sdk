"""Tests for MemoryBank core operations — real embedding + real LLM + ChromaDB."""

import os

import pytest

from reasoning_bank import MemoryBank, MemoryItem


def test_add_and_count(bank):
    assert bank.count() == 0
    bank.add(task_id="t1", query="q1", memory_items=["m1"], status="success")
    assert bank.count() == 1
    bank.add(task_id="t2", query="q2", memory_items=["m2"], status="fail")
    assert bank.count() == 2


def test_list(bank):
    bank.add(task_id="t1", query="q1", memory_items=["m1"])
    bank.add(task_id="t2", query="q2", memory_items=["m2"])
    items = bank.list()
    assert len(items) == 2
    assert {i.task_id for i in items} == {"t1", "t2"}


def test_delete(bank):
    bank.add(task_id="t1", query="q1", memory_items=["m1"])
    bank.add(task_id="t2", query="q2", memory_items=["m2"])
    bank.delete("t1")
    assert bank.count() == 1
    assert bank.list()[0].task_id == "t2"


def test_retrieve(bank):
    bank.add(task_id="t1", query="fix login", memory_items=["use correct button"])
    bank.add(task_id="t2", query="search items", memory_items=["use search bar"])
    results = bank.retrieve(query="fix login", top_k=1)
    assert len(results) == 1


def test_induce(bank, llm_with_retry):
    items = bank.induce(
        task_id="t3",
        query="go to cart",
        trajectory="think...\naction...",
        status="success",
        domain="web",
    )
    assert len(items) >= 1
    assert bank.count() >= 1


def test_induce_scaling(bank, llm_with_retry):
    items = bank.induce_scaling(
        task_id="t4",
        query="find cheapest",
        trajectories=[
            {"trajectory": "traj1...", "status": "success"},
            {"trajectory": "traj2...", "status": "fail"},
        ],
        domain="web",
    )
    assert len(items) >= 1
    assert bank.count() >= 1


def test_induce_requires_llm(tmp_path):
    bank = MemoryBank(
        storage="chroma",
        storage_path=str(tmp_path / "memories"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "gemini"),
    )
    with pytest.raises(ValueError):
        bank.induce(task_id="t", query="q", trajectory="tr", status="success")


def test_memory_item_to_prompt_text():
    item = MemoryItem(
        task_id="t1", query="q1", status="success", domain="web",
        memory_items=["item1", "item2"],
    )
    assert item.to_prompt_text() == "item1\n\nitem2"


def test_memory_item_roundtrip():
    item = MemoryItem(
        task_id="t1", query="q1", status="success", domain="coding",
        memory_items=["mem1"], template_id="tpl-1",
    )
    restored = MemoryItem.from_dict(item.to_dict())
    assert restored.task_id == "t1"
    assert restored.domain == "coding"
    assert restored.memory_items == ["mem1"]
