name = "Дмитрий"
age = 30
height = 1.94
is_studying = True

print(name, type(name))
print(age, type(age))
print(height, type(height))
print(is_studying, type(is_studying))


a = 47
b = 9
print("Сумма:", a + b)
print("Разность:", a - b)
print("Произведение:", a * b)
print("Частное:", a / b)
print("Целочисленное деление:", a // b)
print("Остаток от деления:", a % b)
print("Возведение в степень:", a**b)


raw = "2024"
print(type(raw))
number = int(raw) + 2
print(number, type(number))

# в 4 задаче выведет 5, потому что переменная y ссылается на значение переменной x в момент присваивания. Когда x изменяется на 10, это не влияет на y, так как y уже хранит старое значение 5.
x = 5
y = x
x = 10
print(y)

raw = "2024"

print(raw + "2")  # строка плюс строка
print(int(raw) + 2)  # число плюс число
# print(raw + 2)       # это упадёт
