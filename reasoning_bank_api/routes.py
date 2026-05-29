"""HTTP routes for ReasoningBank REST API — thin wrapper over MemoryBank."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from reasoning_bank import MemoryBank
from reasoning_bank.llm.anthropic_client import AnthropicClient
from reasoning_bank.llm.gemini_client import GeminiClient
from reasoning_bank.llm.openai_client import OpenAIClient
from reasoning_bank.llm.base import LLMClient

router = APIRouter(prefix="/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class AddRequest(BaseModel):
    task_id: str
    query: str
    memory_items: list[str]
    status: str = "success"
    domain: str = "general"
    template_id: str | None = None


class DeleteRequest(BaseModel):
    task_id: str


class InduceRequest(BaseModel):
    task_id: str
    query: str
    trajectory: str
    status: str
    domain: str = "web"


class TrajectoryItem(BaseModel):
    trajectory: str
    status: str


class InduceScalingRequest(BaseModel):
    task_id: str
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
    elif provider in ("vertexai", "google_ai"):
        return GeminiClient(model=model)
    else:
        return OpenAIClient(model=model)


_bank: MemoryBank | None = None


def get_bank() -> MemoryBank:
    global _bank
    if _bank is None:
        _bank = _get_bank()
    return _bank


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/retrieve")
def retrieve(req: RetrieveRequest) -> list[dict]:
    bank = get_bank()
    items = bank.retrieve(query=req.query, top_k=req.top_k)
    return [item.to_dict() for item in items]


@router.post("/add")
def add(req: AddRequest) -> dict:
    bank = get_bank()
    item = bank.add(
        task_id=req.task_id,
        query=req.query,
        memory_items=req.memory_items,
        status=req.status,
        domain=req.domain,
        template_id=req.template_id,
    )
    return {"ok": True, "id": item.id}


@router.post("/delete")
def delete(req: DeleteRequest) -> dict:
    bank = get_bank()
    bank.delete(task_id=req.task_id)
    return {"ok": True}


@router.get("/list")
def list_memories() -> list[dict]:
    bank = get_bank()
    return [item.to_dict() for item in bank.list()]


@router.get("/count")
def count() -> dict:
    bank = get_bank()
    return {"count": bank.count()}


@router.post("/induce")
def induce(req: InduceRequest) -> list[dict]:
    bank = get_bank()
    try:
        items = bank.induce(
            task_id=req.task_id,
            query=req.query,
            trajectory=req.trajectory,
            status=req.status,
            domain=req.domain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [item.to_dict() for item in items]


@router.post("/induce_scaling")
def induce_scaling(req: InduceScalingRequest) -> list[dict]:
    bank = get_bank()
    try:
        items = bank.induce_scaling(
            task_id=req.task_id,
            query=req.query,
            trajectories=[t.model_dump() for t in req.trajectories],
            domain=req.domain,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [item.to_dict() for item in items]
