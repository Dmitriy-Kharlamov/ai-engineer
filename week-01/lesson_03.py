# print("Строка\nВторая строка")   # \n — перенос строки
# print("Колонка1\tКолонка2")      # \t — табуляция
# print("Кавычка: \"текст\"")      # \" — кавычка внутри тех же кавычек
# print("Обратный слеш: \\")       # \\ — сам слеш


text = "  Машинное Обучение и Обработка Естественного Языка  "
print(len(text))  # длина строки
text_2 = text.strip() # удаляем пробелы в начале и конце
text_3 = text_2.lower() # переводим в нижний регистр
print(len(text_3))
words = text_3.split() # разбиваем на слова
print(len(words))


s = "трансформер"
print(s[0])
print(s[-1]) 
print(s[0:4])
print(s[-4:])
print(s[::-1])
print(s[::3])

model = "GPT-4"
accuracy = 0.8734
cost = 1234.5
print(f"Модель: {model}: точность {accuracy:.1%}, стоимость {cost:,.2f} руб.")


raw = "python, java , c++,  rust ,go"
languages = raw.split(",")
languages_2 = [lang.strip() for lang in languages] 
print(languages_2)
line = " | ".join(languages_2)
print(line)


print("Привет".replace("П", "Х"))  # Хривет
# в исходной переменной строка не изменится, так как строки в Python неизменяемы. Метод replace возвращает новую строку с заменой, но не изменяет исходную строку.