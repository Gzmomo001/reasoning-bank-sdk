"""FastAPI application for ReasoningBank REST API."""

from __future__ import annotations

import os

from fastapi import FastAPI

from reasoning_bank_api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="ReasoningBank API",
        version="0.1.0",
        description="Persistent agent memory with induction, retrieval, and scaling.",
    )
    app.include_router(router)
    return app


app = create_app()
