"""Перевод исключений в HTTP-ответы. ragkit не знает про коды — это знает только API."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ragkit import ChunkTooLargeError, EmptyIndexError, RAGError


async def validation_handler(request: Request, exc: Exception) -> JSONResponse:
    """422 без поля input: пароль из тела запроса не уходит обратно клиенту.

    loc, msg, type и ctx остаются — клиент по-прежнему видит, какое поле не прошло и почему.
    """
    assert isinstance(exc, RequestValidationError)
    clean = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
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
