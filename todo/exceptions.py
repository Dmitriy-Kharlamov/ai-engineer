class TodoError(Exception):
    """Базовая ошибка приложения."""


class CorruptedFileError(TodoError):
    def __init__(self, path):
        super().__init__(f"файл {path} повреждён")


class TaskNotFoundError(TodoError):
    def __init__(self, task_id):
        super().__init__(f"задача #{task_id} не найдена")


class InvalidIdError(TodoError):
    def __init__(self, raw_id):
        super().__init__(f"введенное значение не является числом {raw_id} не является числом")


class EmptyTextError(TodoError):
    def __init__(self):
        super().__init__("не передан текст")


class UnknownCommandError(TodoError):
    def __init__(self, command):
        super().__init__(f"передана неизвестная команда {command}")


class TaskAlreadyDoneError(TodoError):
    def __init__(self, task_id):
        super().__init__(f"задача #{task_id} уже сделана")
