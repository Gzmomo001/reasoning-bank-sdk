"""Tests for storage backends."""

import tempfile

from reasoning_bank.core.memory_item import MemoryItem
from reasoning_bank.storage.jsonl import JsonlStorage


def _make_item(query="q1", memory_items=None) -> MemoryItem:
    return MemoryItem(
        query=query,
        status="success",
        domain="web",
        memory_items=memory_items or ["memory text"],
    )


def test_jsonl_add_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        assert store.count() == 0

        store.add(_make_item("q1"))
        assert store.count() == 1

        store.add_batch([_make_item("q2"), _make_item("q3")])
        assert store.count() == 3


def test_jsonl_list():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        store.add(_make_item("q1"))
        store.add(_make_item("q2"))

        items = store.list_all()
        assert len(items) == 2
        assert {i.query for i in items} == {"q1", "q2"}


def test_jsonl_delete():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item1 = _make_item("q1")
        item2 = _make_item("q2")
        store.add(item1)
        store.add(item2)

        store.delete(item1.id)
        assert store.count() == 1
        assert store.list_all()[0].query == "q2"


def test_jsonl_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item = _make_item("fix login", ["use correct button"])
        store.add(item)

        # Store a known embedding keyed by item.id
        emb = [1.0, 0.0, 0.0, 0.0]
        store.store_embedding(item.id, emb)

        results = store.retrieve(emb, top_k=1)
        assert len(results) == 1
        assert results[0].id == item.id


def test_jsonl_cosine_similarity():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item1 = _make_item("q1")
        item2 = _make_item("q2")
        store.add(item1)
        store.add(item2)

        store.store_embedding(item1.id, [1.0, 0.0, 0.0, 0.0])
        store.store_embedding(item2.id, [0.0, 1.0, 0.0, 0.0])

        results = store.retrieve([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].id == item1.id
