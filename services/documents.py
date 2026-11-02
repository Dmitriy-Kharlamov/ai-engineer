"""Документы пользователей. Про HTTP не знает: «чужой» и «нет такого» для него одно и то же."""

from services.db import Session, StoredDocument


class DocumentLimitError(Exception):
    """У владельца уже максимум документов."""


class DocumentRepo:
    def __init__(self, session: Session, max_per_owner: int) -> None:
        self.session = session
        self.max_per_owner = max_per_owner

    def create(self, owner: str, name: str, text: str) -> StoredDocument:
        if self.session.count_by_owner(owner) >= self.max_per_owner:
            raise DocumentLimitError(f"не больше {self.max_per_owner} документов на владельца")
        return self.session.add(owner, name, text)

    def get_owned(self, doc_id: int, owner: str) -> StoredDocument | None:
        """None — и если документа нет, и если он чужой: вызывающий не должен различать."""
        doc = self.session.get(doc_id)
        if doc is None or doc.owner != owner:
            return None
        return doc
