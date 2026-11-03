from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from api.schemas import ConfigOut, HealthOut, ReadyOut
from ragkit import Embedder
from ragkit.config import Settings, get_settings

router = APIRouter(prefix="/health", tags=["health"])

UNKNOWN_VERSION = "unknown"


def ragkit_version() -> str:
    """Версия из метаданных пакета; без `pip install` метаданных нет.

    /health — последний эндпоинт, который имеет право падать: по нему балансировщик
    решает, жив ли сервис. Поэтому отсутствие метаданных — не 500, а "unknown".
    """
    try:
        return version("ragkit")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


@router.get("", summary="Жив ли сервис")
async def health(request: Request) -> HealthOut:
    started_at: datetime = request.app.state.started_at
    return HealthOut(
        status="ok",
        ragkit_version=ragkit_version(),
        started_at=started_at,
    )


@router.get(
    "/ready",
    summary="Готов ли сервис работать (для healthcheck)",
    responses={503: {"model": ReadyOut, "description": "Процесс жив, но работать не может"}},
)
async def ready(request: Request, response: Response) -> ReadyOut:
    """/health отвечает «процесс жив»; /health/ready — «нужная работа выполняется».

    Проверка дешёвая и настоящая: эмбеддер из lifespan считает один вектор, и вектор
    должен быть нужной размерности. Сломанная конфигурация (APP_EMBEDDER_DIM=0) даёт
    живой процесс, который на /health отвечает 200, а работать не может.
    """
    problems: list[str] = []
    embedder: Embedder = request.app.state.embedder
    vectors = embedder.embed(["готов?"])
    if embedder.dim < 1 or len(vectors) != 1 or len(vectors[0]) != embedder.dim:
        problems.append(f"эмбеддер не выдаёт векторы: dim={embedder.dim}")
    if problems:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyOut(ready=not problems, problems=problems)


@router.get("/config", summary="Несекретные настройки")
async def config(settings: Annotated[Settings, Depends(get_settings)]) -> ConfigOut:
    return ConfigOut(
        llm_model=settings.llm.model, debug=settings.debug, max_retries=settings.max_retries
    )
