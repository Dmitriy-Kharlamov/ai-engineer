from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_clock, get_users
from api.schemas import LoginIn, RegisterIn, TokenOut
from api.security import SettingsDep, create_access_token, jwt_secret, unauthorized
from services.clock import Clock
from services.users import UserStore

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201)
def register(body: RegisterIn, users: Annotated[UserStore, Depends(get_users)]) -> dict[str, str]:
    users.register(body.username, body.password.get_secret_value())  # UserExistsError → 409
    return {"username": body.username}


@router.post("/login")
def login(
    body: LoginIn,
    users: Annotated[UserStore, Depends(get_users)],
    secret: Annotated[str, Depends(jwt_secret)],
    settings: SettingsDep,
    clock: Annotated[Clock, Depends(get_clock)],
) -> TokenOut:
    # обычный def: bcrypt блокирует ~0.2 с, FastAPI уведёт это в пул потоков (Ф3-06)
    user = users.authenticate(body.username, body.password.get_secret_value())
    if user is None:
        raise unauthorized("Bearer", "неверный логин или пароль")  # одно сообщение на оба случая
    token = create_access_token(
        user.username, user.roles, secret, settings.jwt_ttl_minutes, now=clock.now()
    )
    return TokenOut(access_token=token, expires_in=settings.jwt_ttl_minutes * 60)
