from .exceptions import EmptyTextError, TaskNotFoundError
from .models import Task


class TaskList:
    def __init__(self, tasks=None, next_id=1):
        self._tasks = tasks if tasks is not None else {}
        self._next_id = next_id

    def add(self, text):
        if not text.strip():
            raise EmptyTextError()
        task = Task(task_id=self._next_id, text=text)
        self._tasks[task.task_id] = task
        self._next_id += 1
        return task
    
    def get(self, task_id):
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        return self._tasks[task_id]

    def complete(self, task_id):
        task = self.get(task_id)
        task.mark_done()
        return task

    def delete(self, task_id):
        task = self.get(task_id)
        del self._tasks[task_id]
        return task

    def filter(self, is_done=None):
        all_tasks = list(self._tasks.values())
        if is_done is None:
            return all_tasks
        return [task for task in all_tasks if task.is_done == is_done]

    def stats(self):
        total = len(self._tasks)
        done = len(self.filter(True))
        percent = round(done / total * 100) if total else 0
        return total, done, percent

    def to_dict(self):
        return {
            "next_id": self._next_id,
            "tasks": [task.to_dict() for task in self.filter()]
        }

    @classmethod
    def from_dict(cls, data):
        tasks = {}
        for task_data in data["tasks"]:
            task = Task.from_dict(task_data)
            tasks[task.task_id] = task
        return cls(tasks=tasks, next_id=data["next_id"])