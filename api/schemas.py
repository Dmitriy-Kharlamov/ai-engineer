"""Модели запросов и ответов. Модель входа, модель базы и модель ответа — разные классы."""

from datetime import datetime

from pydantic import BaseModel, Field, SecretStr

from ragkit import SplitMode

# --- health -----------------------------------------------------------------


class HealthOut(BaseModel):
    status: str = Field(examples=["ok"])
    ragkit_version: str = Field(description="Версия установленного пакета ragkit")
    started_at: datetime = Field(description="Когда сервис стартовал (UTC)")
    embedder_loads: int = Field(
        description="Сколько раз создавался эмбеддер за этот запуск; ровно 1"
    )


class ConfigOut(BaseModel):
    """Несекретная часть настроек. api_key сюда не входит — и не может попасть."""

    llm_model: str
    debug: bool
    max_retries: int


# --- chunks -----------------------------------------------------------------


class SplitIn(BaseModel):
    text: str = Field(min_length=1, description="Текст, который нужно нарезать")
    source: str = Field(examples=["manual.pdf"])
    max_tokens: int = Field(default=50, ge=1, le=1000)
    mode: SplitMode = Field(
        default="words", description="words — по словам, sentences — по предложениям"
    )


class ChunkOut(BaseModel):
    text: str
    source: str
    tokens: int


class DocumentIn(BaseModel):
    name: str
    max_chunk_tokens: int = Field(default=50, ge=1)
    texts: list[str] = Field(description="Тексты чанков; может быть пустым — тогда EmptyIndexError")


class DocumentOut(BaseModel):
    name: str
    chunks: int
    total_tokens: int
    largest: ChunkOut


class EmbedIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class EmbedOut(BaseModel):
    dim: int
    vectors: list[list[float]]


# --- lab: модели для опытов урока -----------------------------------------------


class ParamsBody(BaseModel):
    note: str


class ParamsOut(BaseModel):
    doc_id: int
    limit: int
    x_request_id: str | None
    body: ParamsBody


class UserIn(BaseModel):
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


class Login(BaseModel):
    password: SecretStr = Field(min_length=12)
