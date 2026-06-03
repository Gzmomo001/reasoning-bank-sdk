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


async def test_jsonl_add_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        assert await store.count() == 0

        await store.add(_make_item("q1"))
        assert await store.count() == 1

        await store.add_batch([_make_item("q2"), _make_item("q3")])
        assert await store.count() == 3


async def test_jsonl_list():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        await store.add(_make_item("q1"))
        await store.add(_make_item("q2"))

        items = await store.list_all()
        assert len(items) == 2
        assert {i.query for i in items} == {"q1", "q2"}


async def test_jsonl_delete():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item1 = _make_item("q1")
        item2 = _make_item("q2")
        await store.add(item1)
        await store.add(item2)

        await store.delete(item1.id)
        assert await store.count() == 1
        items = await store.list_all()
        assert items[0].query == "q2"


async def test_jsonl_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item = _make_item("fix login", ["use correct button"])
        await store.add(item)

        # Store a known embedding keyed by item.id
        emb = [1.0, 0.0, 0.0, 0.0]
        await store.store_embedding(item.id, emb)

        results = await store.retrieve(emb, top_k=1)
        assert len(results) == 1
        assert results[0].id == item.id


async def test_jsonl_cosine_similarity():
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlStorage(storage_path=tmp)
        item1 = _make_item("q1")
        item2 = _make_item("q2")
        await store.add(item1)
        await store.add(item2)

        await store.store_embedding(item1.id, [1.0, 0.0, 0.0, 0.0])
        await store.store_embedding(item2.id, [0.0, 1.0, 0.0, 0.0])

        results = await store.retrieve([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].id == item1.id
