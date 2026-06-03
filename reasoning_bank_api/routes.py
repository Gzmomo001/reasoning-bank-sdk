"""HTTP routes for ReasoningBank REST API — RESTful endpoints over MemoryBank."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from reasoning_bank import MemoryBank
from reasoning_bank.llm.anthropic_client import AnthropicClient
from reasoning_bank.llm.gemini_client import GeminiClient
from reasoning_bank.llm.openai_client import OpenAIClient

if TYPE_CHECKING:
    from datetime import datetime

    from reasoning_bank.llm.base import LLMClient

router = APIRouter(prefix="/v1/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Response models — specific per endpoint for accurate OpenAPI docs
# ---------------------------------------------------------------------------


class MemoryItemSchema(BaseModel):
    """Detailed representation of a stored memory item."""

    id: str
    query: str
    status: str
    domain: str
    memory_items: list[str]
    created_at: datetime


class MetaTotal(BaseModel):
    total: int


class CountData(BaseModel):
    count: int


class DeleteData(BaseModel):
    deleted: bool
    id: str


class IdData(BaseModel):
    id: str


class MemoryListResponse(BaseModel):
    data: list[MemoryItemSchema]
    meta: MetaTotal


class SearchResponse(BaseModel):
    data: list[MemoryItemSchema]
    meta: MetaTotal


class CreateItemResponse(BaseModel):
    data: MemoryItemSchema
    meta: dict = {}


class CountResponse(BaseModel):
    data: CountData
    meta: dict = {}


class DeleteResponse(BaseModel):
    data: DeleteData
    meta: dict = {}


class InductionResponse(BaseModel):
    data: list[MemoryItemSchema]
    meta: MetaTotal


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AddRequest(BaseModel):
    query: str
    memory_items: list[str]
    status: str = "success"
    domain: str = "general"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class InduceRequest(BaseModel):
    query: str
    trajectory: str
    status: str
    domain: str = "web"


class TrajectoryItem(BaseModel):
    trajectory: str
    status: str


class InduceBatchRequest(BaseModel):
    query: str
    trajectories: list[TrajectoryItem]
    domain: str = "web"


# ---------------------------------------------------------------------------
# Dependency: shared MemoryBank instance
# ---------------------------------------------------------------------------


def _get_bank() -> MemoryBank:
    """Create a MemoryBank from environment configuration."""
    storage = os.environ.get("STORAGE", "chroma")
    storage_path = os.environ.get("STORAGE_PATH", "./memories")
    emb_provider = os.environ.get("EMBEDDING_PROVIDER", "gemini")
    emb_model = os.environ.get("EMBEDDING_MODEL")
    llm = _get_llm()
    return MemoryBank(
        storage=storage,
        storage_path=storage_path,
        embedding_provider=emb_provider,
        embedding_model=emb_model,
        llm_client=llm,
    )


def _get_llm() -> LLMClient | None:
    provider = os.environ.get("LLM_PROVIDER", "")
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        return None
    if provider == "anthropic":
        return AnthropicClient(model=model)
    if provider in ("vertexai", "google_ai"):
        return GeminiClient(model=model)
    return OpenAIClient(model=model)


_bank: MemoryBank | None = None


def get_bank() -> MemoryBank:
    global _bank  # noqa: PLW0603
    if _bank is None:
        _bank = _get_bank()
    return _bank


# ---------------------------------------------------------------------------
# Routes — Memory Items CRUD
# ---------------------------------------------------------------------------


@router.get("/items", response_model=MemoryListResponse)
def list_items() -> MemoryListResponse:
    """List all stored memories."""
    bank = get_bank()
    items = bank.list()
    return MemoryListResponse(
        data=[MemoryItemSchema(**item.to_dict()) for item in items],
        meta=MetaTotal(total=len(items)),
    )


@router.get("/items/count", response_model=CountResponse)
def count_items() -> CountResponse:
    """Get total number of stored memories."""
    bank = get_bank()
    return CountResponse(data=CountData(count=bank.count()))


@router.get("/items/search", response_model=SearchResponse)
def search_items_get(
    query: Annotated[str, Query(description="Search query text")],
    top_k: Annotated[int, Query(ge=1, description="Number of results to return")] = 3,
) -> SearchResponse:
    """Retrieve relevant memories (simple query via GET params)."""
    bank = get_bank()
    items = bank.retrieve(query=query, top_k=top_k)
    return SearchResponse(
        data=[MemoryItemSchema(**item.to_dict()) for item in items],
        meta=MetaTotal(total=len(items)),
    )


@router.post("/items/search", response_model=SearchResponse)
def search_items_post(req: SearchRequest) -> SearchResponse:
    """Retrieve relevant memories (complex query via POST body)."""
    bank = get_bank()
    items = bank.retrieve(query=req.query, top_k=req.top_k)
    return SearchResponse(
        data=[MemoryItemSchema(**item.to_dict()) for item in items],
        meta=MetaTotal(total=len(items)),
    )


@router.post("/items", status_code=201, response_model=CreateItemResponse)
def create_item(req: AddRequest) -> CreateItemResponse:
    """Add a memory item directly."""
    bank = get_bank()
    item = bank.add(
        query=req.query,
        memory_items=req.memory_items,
        status=req.status,
        domain=req.domain,
    )
    return CreateItemResponse(data=MemoryItemSchema(**item.to_dict()))


@router.delete("/items/{item_id}", response_model=DeleteResponse)
def delete_item(item_id: str) -> DeleteResponse:
    """Delete a memory item by its ID."""
    bank = get_bank()
    bank.delete(item_id=item_id)
    return DeleteResponse(data=DeleteData(deleted=True, id=item_id))


# ---------------------------------------------------------------------------
# Routes — Inductions
# ---------------------------------------------------------------------------


@router.post("/inductions", status_code=201, response_model=InductionResponse)
def create_induction(req: InduceRequest) -> InductionResponse:
    """Run single-trajectory auto induction using LLM."""
    bank = get_bank()
    try:
        items = bank.induce(
            query=req.query,
            trajectory=req.trajectory,
            status=req.status,
            domain=req.domain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return InductionResponse(
        data=[MemoryItemSchema(**item.to_dict()) for item in items],
        meta=MetaTotal(total=len(items)),
    )


@router.post("/inductions/batch", status_code=201, response_model=InductionResponse)
def create_induction_batch(req: InduceBatchRequest) -> InductionResponse:
    """Run multi-trajectory contrast induction."""
    bank = get_bank()
    try:
        items = bank.induce_scaling(
            query=req.query,
            trajectories=[t.model_dump() for t in req.trajectories],
            domain=req.domain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return InductionResponse(
        data=[MemoryItemSchema(**item.to_dict()) for item in items],
        meta=MetaTotal(total=len(items)),
    )
