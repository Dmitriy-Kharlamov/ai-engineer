from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_repo
from api.schemas import DocumentCreateIn, StoredDocumentOut
from api.security import current_user
from services.apikeys import Principal
from services.documents import DocumentRepo

router = APIRouter(prefix="/documents", tags=["documents"])

UserDep = Annotated[Principal, Depends(current_user)]
RepoDep = Annotated[DocumentRepo, Depends(get_repo)]


@router.post("", status_code=201)
def create_document(body: DocumentCreateIn, user: UserDep, repo: RepoDep) -> StoredDocumentOut:
    doc = repo.create(user.name, body.name, body.text)
    return StoredDocumentOut.model_validate(doc)


@router.get("/{doc_id}")
def get_document(doc_id: int, user: UserDep, repo: RepoDep) -> StoredDocumentOut:
    doc = repo.get_owned(doc_id, user.name)
    if doc is None:
        # и «нет такого», и «чужой» — одинаковый 404: не подтверждаем, что id существует
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="документ не найден")
    return StoredDocumentOut.model_validate(doc)
