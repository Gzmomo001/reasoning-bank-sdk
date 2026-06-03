"""Tests for MemoryBank core operations — real embedding + real LLM + ChromaDB."""

import os

import pytest

from reasoning_bank import MemoryBank, MemoryItem


def test_add_and_count(bank):
    assert bank.count() == 0
    bank.add(query="q1", memory_items=["m1"], status="success")
    assert bank.count() == 1
    bank.add(query="q2", memory_items=["m2"], status="fail")
    assert bank.count() == 2


def test_list(bank):
    bank.add(query="q1", memory_items=["m1"])
    bank.add(query="q2", memory_items=["m2"])
    items = bank.list()
    assert len(items) == 2
    assert {i.query for i in items} == {"q1", "q2"}


def test_delete(bank):
    item1 = bank.add(query="q1", memory_items=["m1"])
    bank.add(query="q2", memory_items=["m2"])
    bank.delete(item1.id)
    assert bank.count() == 1
    assert bank.list()[0].query == "q2"


def test_retrieve(bank):
    bank.add(query="fix login", memory_items=["use correct button"])
    bank.add(query="search items", memory_items=["use search bar"])
    results = bank.retrieve(query="fix login", top_k=1)
    assert len(results) == 1


def test_induce(bank, llm_with_retry):
    items = bank.induce(
        query="go to cart",
        trajectory="think...\naction...",
        status="success",
        domain="web",
    )
    assert len(items) >= 1
    assert bank.count() >= 1


def test_induce_scaling(bank, llm_with_retry):
    items = bank.induce_scaling(
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
        bank.induce(query="q", trajectory="tr", status="success")


def test_memory_item_to_prompt_text():
    item = MemoryItem(
        query="q1",
        status="success",
        domain="web",
        memory_items=["item1", "item2"],
    )
    assert item.to_prompt_text() == "item1\n\nitem2"


def test_memory_item_roundtrip():
    item = MemoryItem(
        query="q1",
        status="success",
        domain="coding",
        memory_items=["mem1"],
    )
    restored = MemoryItem.from_dict(item.to_dict())
    assert restored.id == item.id
    assert restored.domain == "coding"
    assert restored.memory_items == ["mem1"]
