name = input("Как вас зовут? ")
birth_year = int(input("В каком году вы родились? "))
print(f"Привет, {name}! В 2026 году вам будет {2026 - birth_year} лет.")


raw = input("Введите два числа через пробел: ")
parts = raw.split()

a = int(parts[0])
b = int(parts[1])

print(f"Сумма: {a + b}")
print(f"Разность: {a - b}")
print(f"Произведение: {a * b}")
print(f"Частное: {a / b:.2f}")

raw = input("Введите 4 числа через пробел: ")
parts = raw.split()

print(f"Сумма: {sum(int(x) for x in parts)}", f"Количество: {len(parts)}")


raw = input("Введите цену товара и количество через пробел: ")
parts = raw.split()

price = float(parts[0])
quantity = int(parts[1])

print(f"{quantity} шт. * {price:,.2f} руб. = {price * quantity:,.2f} руб.")

# Если вторым числом будет 0, то при делении на ноль возникнет ошибка
