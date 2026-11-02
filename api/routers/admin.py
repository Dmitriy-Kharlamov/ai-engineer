"""Админка. Защищена на уровне роутера: каждый эндпоинт здесь требует ключ с ролью admin,
даже если в его подписи защита не упомянута."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_api_keys, get_database
from api.schemas import DbStatsOut, KeyIssueIn, KeyIssueOut
from api.security import require_admin
from services.apikeys import ApiKeyStore
from services.db import Database

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats")
def stats(db: Annotated[Database, Depends(get_database)]) -> DbStatsOut:
    s = db.stats
    return DbStatsOut(
        documents=len(db.documents),
        sessions_opened=s.opened,
        commits=s.committed,
        rollbacks=s.rolled_back,
        closed=s.closed,
    )


@router.post("/keys", status_code=201)
def issue_key(
    body: KeyIssueIn, store: Annotated[ApiKeyStore, Depends(get_api_keys)]
) -> KeyIssueOut:
    return KeyIssueOut(name=body.name, key=store.issue(body.name, frozenset(body.roles)))


@router.get("/new-report")
def new_report() -> dict[str, str]:
    # Добавлен последним. Ни Depends, ни проверки ключа здесь нет — защищает сам роутер.
    return {"report": "секретная сводка"}
