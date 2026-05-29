"""Tests for storage backends."""

import tempfile

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.storage.jsonl import JsonlStorage


def _make_item(task_id="t1", query="q1", memory_items=None) -> MemoryItem:
    return MemoryItem(
        task_id=task_id,
        query=query,
        status="success",
        domain="web",
        memory_items=memory_items or ["memory text"],
    )


def test_jsonl_add_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        assert store.count() == 0

        store.add(_make_item("t1"))
        assert store.count() == 1

        store.add_batch([_make_item("t2"), _make_item("t3")])
        assert store.count() == 3


def test_jsonl_list():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        store.add(_make_item("t1", "q1"))
        store.add(_make_item("t2", "q2"))

        items = store.list_all()
        assert len(items) == 2
        assert {i.task_id for i in items} == {"t1", "t2"}


def test_jsonl_delete():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        store.add(_make_item("t1"))
        store.add(_make_item("t2"))

        store.delete("t1")
        assert store.count() == 1
        assert store.list_all()[0].task_id == "t2"


def test_jsonl_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item = _make_item("t1", "fix login", ["use correct button"])
        store.add(item)

        # Store a known embedding
        emb = [1.0, 0.0, 0.0, 0.0]
        store.store_embedding("t1", emb)

        results = store.retrieve(emb, top_k=1)
        assert len(results) == 1
        assert results[0].task_id == "t1"


def test_jsonl_cosine_similarity():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        store.add(_make_item("t1"))
        store.add(_make_item("t2"))

        store.store_embedding("t1", [1.0, 0.0, 0.0, 0.0])
        store.store_embedding("t2", [0.0, 1.0, 0.0, 0.0])

        results = store.retrieve([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].task_id == "t1"
