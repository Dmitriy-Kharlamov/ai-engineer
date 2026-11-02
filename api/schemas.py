"""Модели запросов и ответов. Модель входа, модель базы и модель ответа — разные классы."""

from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr
from pydantic_core import PydanticCustomError

from ragkit import SplitMode


class InputModel(BaseModel):
    """База для моделей ВХОДА: опечатка в имени поля — 422, а не молчаливое умолчание."""

    model_config = ConfigDict(extra="forbid")


def check_source_name(value: str) -> str:
    """Имя источника: без пустоты и без путей (../, /), чтобы его можно было класть в файл."""
    value = value.strip()
    if not value:
        raise PydanticCustomError("empty_source", "имя источника не может быть пустым")
    if "/" in value or "\\" in value or ".." in value:
        raise PydanticCustomError("source_path", "имя источника не должно содержать путь")
    return value


# Правило как тип: одно определение, две модели (SplitIn.source и DocumentIn.name).
SourceName = Annotated[str, AfterValidator(check_source_name)]

# --- health -----------------------------------------------------------------


class HealthOut(BaseModel):
    status: str = Field(examples=["ok"])
    ragkit_version: str = Field(description="Версия установленного пакета ragkit")
    started_at: datetime = Field(description="Когда сервис стартовал (UTC)")


class ConfigOut(BaseModel):
    """Несекретная часть настроек. api_key сюда не входит — и не может попасть."""

    llm_model: str
    debug: bool
    max_retries: int


# --- chunks -----------------------------------------------------------------


class SplitIn(InputModel):
    text: str = Field(min_length=1, description="Текст, который нужно нарезать")
    source: SourceName = Field(examples=["manual.pdf"])
    max_tokens: int = Field(default=50, ge=1, le=1000)
    mode: SplitMode = Field(
        default="words", description="words — по словам, sentences — по предложениям"
    )


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # ChunkOut.model_validate(chunk)

    text: str
    source: str
    tokens: int


class DocumentIn(InputModel):
    name: SourceName
    max_chunk_tokens: int = Field(default=50, ge=1)
    texts: list[str] = Field(description="Тексты чанков; может быть пустым — тогда EmptyIndexError")


class DocumentOut(BaseModel):
    name: str
    chunks: int
    total_tokens: int
    largest: ChunkOut


class EmbedIn(InputModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class EmbedOut(BaseModel):
    dim: int
    vectors: list[list[float]]


# --- lab: модели для опытов урока -----------------------------------------------


class ParamsBody(InputModel):
    note: str


class ParamsOut(BaseModel):
    doc_id: int
    limit: int
    x_request_id: str | None
    body: ParamsBody


class UserIn(InputModel):
    """Что присылает клиент."""

    name: str
    password: SecretStr


class UserDB(BaseModel):
    """Что лежит в базе. Наружу целиком не отдаётся никогда."""

    name: str
    password_hash: str
    internal_id: int


class UserOut(BaseModel):
    """Что видит клиент."""

    name: str


class UserOutStrict(BaseModel):
    """Модель ответа с полем, которого во внутренней модели нет."""

    name: str
    email: str


class Login(InputModel):
    password: SecretStr = Field(min_length=12)


# --- auth, документы, админка (Ф3-08) -------------------------------------------


class RegisterIn(InputModel):
    username: SourceName = Field(min_length=3, max_length=32)
    password: SecretStr = Field(min_length=8)


class LoginIn(InputModel):
    username: str
    password: SecretStr


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Секунд до истечения")


class PrincipalOut(BaseModel):
    name: str
    roles: list[str]


class DocumentCreateIn(InputModel):
    name: SourceName
    text: str = Field(min_length=1)


class StoredDocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    text: str


class KeyIssueIn(InputModel):
    name: str = Field(min_length=1)
    roles: list[str] = Field(default_factory=list)


class KeyIssueOut(BaseModel):
    name: str
    key: str = Field(description="Показывается один раз; на сервере хранится только sha256")


class DbStatsOut(BaseModel):
    documents: int
    sessions_opened: int
    commits: int
    rollbacks: int
    closed: int
