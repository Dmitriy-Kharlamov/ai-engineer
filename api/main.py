"""Точка входа: uvicorn api.main:app --reload"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from api import deps
from api.errors import install_error_handlers
from api.routers import chunks, health, lab


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.started_at = datetime.now(UTC)
    app.state.loads_before = deps.EMBEDDER_LOADS
    app.state.embedder = deps.load_embedder(dim=128)  # один раз за запуск, не на запрос
    yield
    # здесь закрывают то, что открыли: пулы соединений, клиенты, файлы


app = FastAPI(title="ragkit API", version="0.1.0", lifespan=lifespan)
install_error_handlers(app)
app.include_router(health.router)
app.include_router(chunks.router)
app.include_router(lab.router)
