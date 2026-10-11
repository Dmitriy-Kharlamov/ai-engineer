from dataclasses import dataclass, field
from datetime import datetime

from .exceptions import TaskAlreadyDoneError


@dataclass
class Task:
    task_id: int
    text: str
    is_done: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    done_at: datetime | None = None
  
    def mark_done(self):
        if self.is_done:
            raise TaskAlreadyDoneError(self.task_id)
        self.is_done = True
        self.done_at = datetime.now()

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "text": self.text,
            "is_done": self.is_done,
            "created_at": self.created_at.isoformat(),
            "done_at": self.done_at.isoformat() if self.done_at else None
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            task_id=data["task_id"],
            text=data["text"],
            is_done=data["is_done"],
            created_at=datetime.fromisoformat(data["created_at"]),
            done_at=datetime.fromisoformat(data["done_at"]) if data["done_at"] else None
        )

    