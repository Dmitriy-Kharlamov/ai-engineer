for i in range(1, 11):
    print(i)


total = 0
for n in range(1, 101):
    total = total + n
print(total)


words = ["трансформер", "токен", "эмбеддинг", "attention", "слой"]
for i, word in enumerate(words, start=1):
    print(f"{i}. {word}")


for word in words:
    if len(word) <= 5:
        continue
    print(word)


raw = input("Введите числа через пробел: ")
numbers = [int(x) for x in raw.split()]
if not numbers:
    print("Вы не ввели ни одного числа")
else:
    total = 0
    largest = numbers[0]
    minimum = numbers[0]
    for n in numbers:
        total += n
        if n > largest:
            largest = n
        if n < minimum:
            minimum = n

    print(f"Сумма: {total}, наибольшее: {largest}, наименьшее: {minimum}")


i = 0
while True:
    raw = int(input("Угадайте число: "))
    i += 1
    if raw > 42:
        print(f"Вы ввели {raw}, это больше загаданного числа")
    elif raw < 42:
        print(f"Вы ввели {raw}, это меньше загаданного числа")
    elif raw == 42:
        break
print(f"Вы угадали! Использовано {i} попыток.")


for i in range(2, 6):
    for j in range(1, 11):
        print(f"{i} × {j} = {i * j}")


# Итогом будет 0 0, 1 0 и 2 0
