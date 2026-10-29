from datetime import datetime
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from api import deps
from api.schemas import ConfigOut, HealthOut
from ragkit.config import Settings, get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Жив ли сервис")
async def health(request: Request) -> HealthOut:
    started_at: datetime = request.app.state.started_at
    loads_before: int = request.app.state.loads_before
    return HealthOut(
        status="ok",
        ragkit_version=version("ragkit"),
        started_at=started_at,
        embedder_loads=deps.EMBEDDER_LOADS - loads_before,
    )


@router.get("/config", summary="Несекретные настройки")
async def config(settings: Annotated[Settings, Depends(get_settings)]) -> ConfigOut:
    return ConfigOut(
        llm_model=settings.llm.model, debug=settings.debug, max_retries=settings.max_retries
    )
