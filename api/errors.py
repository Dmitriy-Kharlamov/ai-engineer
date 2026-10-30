"""Перевод исключений в HTTP-ответы. ragkit не знает про коды — это знает только API."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ragkit import ChunkTooLargeError, EmptyIndexError, RAGError

# Белый список, а не чёрный: в ответ попадает только то, что перечислено.
SAFE_ERROR_KEYS = ("type", "loc", "msg")
SAFE_CTX_KEYS = frozenset(
    {"min_length", "max_length", "actual_length", "ge", "gt", "le", "lt", "expected", "pattern"}
)


def safe_error(err: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: err[key] for key in SAFE_ERROR_KEYS if key in err}
    ctx = {k: v for k, v in err.get("ctx", {}).items() if k in SAFE_CTX_KEYS}
    if ctx:
        out["ctx"] = ctx
    return out


async def validation_handler(request: Request, exc: Exception) -> JSONResponse:
    """422 только из безопасных полей: ни input, ни произвольный ctx клиенту не уходят.

    Клиент по-прежнему видит loc, type, msg и числовые ограничения (min_length и т. п.).
    Против секрета внутри msg белый список бессилен — это забота валидатора
    (PydanticCustomError без интерполяции значения).
    """
    assert isinstance(exc, RequestValidationError)
    clean = [safe_error(err) for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(clean)})


Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def _ragkit_error(status_code: int) -> Handler:
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"detail": str(exc), "error": type(exc).__name__},
        )

    return handler


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_handler)
    # от частного к общему: FastAPI ищет обработчик по MRO исключения
    app.add_exception_handler(ChunkTooLargeError, _ragkit_error(413))
    app.add_exception_handler(EmptyIndexError, _ragkit_error(422))
    app.add_exception_handler(RAGError, _ragkit_error(400))
