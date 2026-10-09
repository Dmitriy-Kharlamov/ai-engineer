#1
def total(*numbers):
    return sum(numbers) 

print(total(0, 2, 3, 4, 5))


# 2
def describe_model(name, **specs):
    print(f"Модель: {name}")
    for key, value in specs.items():
        print(f"  {key}: {value}")
describe_model("iPhone", color="black", storage="128GB")


# 3 
def build_prompt(template, **values):
    return template.format(**values)

print(build_prompt("Переведи {text} на {language}", text="Hello, world!", language="русский"))


# 4
numbers = [3, 7, 2, 9]
max_number = max(*numbers)
print(f"Наибольшее число: {max_number}")

max_number = max(numbers)
print(f"Наибольшее число: {max_number}")
# max(iterable)        # один аргумент-коллекция → ищет максимум внутри неё
# max(a, b, c)         # несколько аргументов → ищет максимум среди них
# для коллекции пишите max(numbers), без звёздочки. Распаковка здесь лишняя и создаёт лишний кортеж в памяти. Звёздочка нужна, когда функция действительно ждёт отдельные аргументы — как zip(*results)



# 5
def make_config(**overrides):
    default_config = {"model": "claude-opus-5", "temperature": 0.7, "max_tokens": 1000}
    default_config.update(overrides)
    return default_config
print(make_config(temperature=0.9, max_tokens=500))


# 6
def apply_to_all(func, *items):
    return [func(item) for item in items]

print(apply_to_all(len, "hello", "world"))
print(apply_to_all(str.upper, "hello", "world"))
print(apply_to_all(lambda x: x * 2, 1, 2, 3))


# 7
x = 10

def show():
    return x

def change():
    x = 20
    return x

print(show()) # Данная функция выводит значение глобальной переменной x, которая равна 10. Она не изменяет значение x, а просто выводит его.
print(change()) # Данная функция создает локальную переменную x внутри функции и присваивает ей значение 20. Она выводит это локальное значение, но не изменяет глобальную переменную x. Поэтому после вызова change() глобальная переменная x остается равной 10   
print(show()) # Данная функция снова выводит значение глобальной переменной x, которая по-прежнему равна 10, так как вызов функции change() не изменил глобальную переменную x.


# 8
sum_1 = 0 #Нельзя использовать встроенные функции как имена переменных, так как это может привести к неожиданным ошибкам в коде. Например, если вы присвоите значение переменной с именем sum, то вы потеряете доступ к встроенной функции sum() и не сможете использовать её в дальнейшем. Поэтому лучше использовать другие имена для переменных, чтобы избежать конфликтов с встроенными функциями.
numbers = [1, 2, 3]
print(sum(numbers))