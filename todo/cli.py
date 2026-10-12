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
    while True:
        line = input("> ").strip()
        if not line:
            print("Менеджер задач. Введите help для списка команд.")
            continue
        if line == "exit":
            print("Всего доброго")
            break
        if line == "help":
            print_help()
        else:
            print(f"Неизвестная команда: {line}")