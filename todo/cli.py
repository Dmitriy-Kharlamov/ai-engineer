from .exceptions import (
    CorruptedFileError,
    InvalidIdError,
    TodoError,
    UnknownCommandError,
)
from .storage import TaskStorage
from .task_list import TaskList

DATA_FILE = "tasks.json"


def load_task_list(storage):
    if not storage.exists():
        print("Файл с данными не найден, начинаем с пустого списка")
        return TaskList()
    try:
        return storage.load()
    except CorruptedFileError as error:
        print(f"Ошибка: {error}. Начинаем с пустого списка.")
        return TaskList()


def handle_add(task_list, argument):
    task = task_list.add(argument)
    print(f"Задача #{task.task_id} добавлена")


def parse_line(line):
    parts = line.split(maxsplit=1)
    command = parts[0]
    argument = parts[1] if len(parts) > 1 else ""
    return command, argument


def print_help():
    print("Команды:")
    print("add <текст> — добавить задачу")
    print("list — показать все задачи")
    print("list done — показать выполненные")
    print("list todo — показать невыполненные")
    print("done <id> — отметить задачу выполненной")
    print("delete <id> — удалить задачу")
    print("stats — статистика")
    print("help — эта справка")
    print("exit — сохранить и выйти")


def format_task(task):
    mark = "[x]" if task.is_done else "[ ]"
    return f"#{task.task_id:<3} {mark}  {task.text:<40} {task.created_at:%d.%m.%Y}"


def handle_list(task_list, argument):
    if argument == "":
        tasks = task_list.filter()
    elif argument == "done":
        tasks = task_list.filter(True)
    elif argument == "todo":
        tasks = task_list.filter(False)
    else:
        raise UnknownCommandError(f"list {argument}")
    if not tasks:
        print("Список пуст")
    for task in tasks:
        print(format_task(task))


def parse_id(argument):
    if not argument.isdigit():
        raise InvalidIdError(argument)
    return int(argument)


def handle_done(task_list, argument):
    task = task_list.complete(parse_id(argument))
    print(f"Задача #{task.task_id} выполнена")


def handle_delete(task_list, argument):
    task = task_list.delete(parse_id(argument))
    print(f"Задача #{task.task_id} удалена")


def handle_stats(task_list):
    total, done, percent = task_list.stats()
    print(f"Всего: {total}, выполнено: {done} ({percent}%)")


def run():
    print("Менеджер задач. Введите help для списка команд.")
    storage = TaskStorage(DATA_FILE)
    task_list = load_task_list(storage)
    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            line = "exit"
        if not line:
            print("Менеджер задач. Введите help для списка команд.")
            continue
        command, argument = parse_line(line)
        if command == "exit":
            storage.save(task_list)
            print("Сохранено. До встречи.")
            break
        try:
            if command == "help":
                print_help()
            elif command == "add":
                handle_add(task_list, argument)
            elif command == "list":
                handle_list(task_list, argument)
            elif command == "done":
                handle_done(task_list, argument)
            elif command == "delete":
                handle_delete(task_list, argument)
            elif command == "stats":
                handle_stats(task_list)
            else:
                raise UnknownCommandError(command)
        except TodoError as error:
            print(f"Ошибка: {error}")
