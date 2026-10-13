import json
import os

from .exceptions import CorruptedFileError
from .task_list import TaskList


class TaskStorage:
    def __init__(self, path):
        self.path = path

    def exists(self):
        return os.path.exists(self.path)
              
    def load(self):
        if not self.exists():
            return TaskList()                      
        try:
            with open(self.path, encoding="utf-8") as file:
                data = json.load(file)                  
            return TaskList.from_dict(data)                      
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            raise CorruptedFileError(self.path)
        
    def save(self, task_list):
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(task_list.to_dict(), file, ensure_ascii=False, indent=2)