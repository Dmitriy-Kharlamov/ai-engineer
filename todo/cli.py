from .exceptions import TodoError, UnknownCommandError
from .task_list import TaskList


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


def run():
    print("Менеджер задач. Введите help для списка команд.")
    task_list = TaskList()
    while True:
        line = input("> ").strip()
        if not line:
            print("Менеджер задач. Введите help для списка команд.")
            continue
        command, argument = parse_line(line)
        if command == "exit":
            print("Всего доброго")
            break
        try:
            if command == "help":
                print_help()
            elif command == "add":
                handle_add(task_list, argument)
            else:
                raise UnknownCommandError(command)
        except TodoError as error:
            print(f"Ошибка: {error}")