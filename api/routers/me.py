from typing import Annotated

from fastapi import APIRouter, Depends

from api.schemas import PrincipalOut
from api.security import current_user, rate_limited
from services.apikeys import Principal

router = APIRouter(prefix="/me", tags=["me"])


@router.get("", summary="Кто я (по JWT)")
def me(user: Annotated[Principal, Depends(current_user)]) -> PrincipalOut:
    return PrincipalOut(name=user.name, roles=sorted(user.roles))


@router.get("/key", summary="Кто я (по API-ключу), с ограничением частоты")
def me_by_key(principal: Annotated[Principal, Depends(rate_limited)]) -> PrincipalOut:
    return PrincipalOut(name=principal.name, roles=sorted(principal.roles))
