"""FastAPI application for ReasoningBank REST API."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from reasoning_bank.logging_config import setup_logging
from reasoning_bank_api.routes import router

# Load .env from project root (walk up to find it)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

setup_logging("api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ReasoningBank API",
        version="0.1.0",
        description="Persistent agent memory with induction, retrieval, and scaling.",
    )
    app.include_router(router)

    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        """Lightweight health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
