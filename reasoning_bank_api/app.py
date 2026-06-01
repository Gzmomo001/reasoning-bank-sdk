"""FastAPI application for ReasoningBank REST API."""

from __future__ import annotations

import os

from fastapi import FastAPI

from reasoning_bank.logging_config import setup_logging
from reasoning_bank_api.routes import router

setup_logging("api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ReasoningBank API",
        version="0.1.0",
        description="Persistent agent memory with induction, retrieval, and scaling.",
    )
    app.include_router(router)
    return app


app = create_app()
