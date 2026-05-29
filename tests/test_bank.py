"""Tests for MemoryBank core operations."""

import tempfile
import os

from reasoning_bank import MemoryBank, MemoryItem
from reasoning_bank.core.embedding import CustomEmbedding
from reasoning_bank.llm.base import LLMClient
from reasoning_bank.storage.jsonl import JsonlStorage


class _DummyLLM(LLMClient):
    """LLM that returns a fixed response for testing."""

    def __init__(self, response: str = "# Memory Item 1\n## Title Test\n## Content test content"):
        self._response = response

    def chat(self, messages, system=None):
        return self._response


class _DummyEmbedding(CustomEmbedding):
    """Deterministic embedding for testing."""

    def __init__(self, dim: int = 8):
        super().__init__(self._fake_embed, dim=dim)
        self._counter = 0

    def _fake_embed(self, texts):
        results = []
        for text in texts:
            self._counter += 1
            results.append([float(self._counter)] * self._dim)
        return results


def _make_bank(tmp: str) -> MemoryBank:
    return MemoryBank(
        storage="jsonl",
        storage_path=tmp,
        embedding_provider=_DummyEmbedding(),
    )


def test_add_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _make_bank(tmp)
        assert bank.count() == 0

        bank.add(task_id="t1", query="q1", memory_items=["m1"], status="success")
        assert bank.count() == 1

        bank.add(task_id="t2", query="q2", memory_items=["m2"], status="fail")
        assert bank.count() == 2


def test_list():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _make_bank(tmp)
        bank.add(task_id="t1", query="q1", memory_items=["m1"])
        bank.add(task_id="t2", query="q2", memory_items=["m2"])

        items = bank.list()
        assert len(items) == 2
        assert {i.task_id for i in items} == {"t1", "t2"}


def test_delete():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _make_bank(tmp)
        bank.add(task_id="t1", query="q1", memory_items=["m1"])
        bank.add(task_id="t2", query="q2", memory_items=["m2"])

        bank.delete("t1")
        assert bank.count() == 1
        assert bank.list()[0].task_id == "t2"


def test_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _make_bank(tmp)
        bank.add(task_id="t1", query="fix login", memory_items=["use correct button"])
        bank.add(task_id="t2", query="search items", memory_items=["use search bar"])

        results = bank.retrieve(query="fix login", top_k=1)
        assert len(results) == 1


def test_induce():
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(
            storage="jsonl",
            storage_path=tmp,
            embedding_provider=_DummyEmbedding(),
            llm_client=_DummyLLM("# Memory Item 1\n## Title Nav\n## Content click nav bar"),
        )
        items = bank.induce(
            task_id="t3",
            query="go to cart",
            trajectory="think...\naction...",
            status="success",
            domain="web",
        )
        assert len(items) >= 1
        assert bank.count() >= 1


def test_induce_scaling():
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(
            storage="jsonl",
            storage_path=tmp,
            embedding_provider=_DummyEmbedding(),
            llm_client=_DummyLLM("# Memory Item 1\n## Title Pattern\n## Content use pattern"),
        )
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


def test_induce_requires_llm():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _make_bank(tmp)
        try:
            bank.induce(task_id="t", query="q", trajectory="tr", status="success")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def test_memory_item_to_prompt_text():
    item = MemoryItem(
        task_id="t1",
        query="q1",
        status="success",
        domain="web",
        memory_items=["item1", "item2"],
    )
    assert item.to_prompt_text() == "item1\n\nitem2"


def test_memory_item_roundtrip():
    item = MemoryItem(
        task_id="t1",
        query="q1",
        status="success",
        domain="coding",
        memory_items=["mem1"],
        template_id="tpl-1",
    )
    d = item.to_dict()
    restored = MemoryItem.from_dict(d)
    assert restored.task_id == "t1"
    assert restored.domain == "coding"
    assert restored.memory_items == ["mem1"]
