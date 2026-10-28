# Задача 1:
number = int(input("Число: "))
if number == 0:
    print("Число равно 0")
elif number > 0:
    print("Число положительное")
else:
    print("Число отрицательное")

# Задача 2:
year = int(input("Год: "))
if year % 4 == 0 and year % 100 != 0:
    print("Год високосный (деление на 4 и запрет деления на 100)")
elif year % 400 == 0:
    print("Год високосный (деление на 400)")
else:
    print("Год не високосный")

# Задача 3: Спросите три числа через пробел. Выведите наибольшее из них. Функцией max() пользоваться нельзя — только через if.
numbers = input("Введите 3 числа через пробел: ")
parts = numbers.split()

a = int(parts[0])
b = int(parts[1])
c = int(parts[2])

if a >= b and a >= c:
    largest = a
elif b >= c:
    largest = b
else:
    largest = c

print(f"Наибольшее: {largest}")


# Задача 4:
a = float(input("Первое число: "))
operation = input("Операция (+, -, *, /): ")
b = float(input("Второе число: "))
result = None
if operation == "+":
    result = a + b
elif operation == "-":
    result = a - b
elif operation == "*":
    result = a * b
elif operation == "/" and b == 0:
    print("На ноль делить нельзя")
elif operation == "/":
    result = a / b
else:
    print(f"Неизвестная операция: {operation}")

if result is not None:
    print(f"Результат: {result}")


# Задача 5:
age = int(input("Возраст: "))

if age < 0 or age > 120:
    print("Некорректный возраст")
elif age <= 12:
    print("Ребёнок")
elif age <= 17:
    print("Подросток")
elif age <= 64:
    print("Взрослый")
else:
    print("Пожилой")


# Задача 6:
# Выведется значение B, потому что 0 в if x будет ложью
