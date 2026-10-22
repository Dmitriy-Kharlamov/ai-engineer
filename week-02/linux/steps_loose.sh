#!/usr/bin/env bash
# без set -euo pipefail — так писать НЕ надо, это демонстрация

echo "шаг 1: начинаю"
cat /no/such/file.txt          # шаг 2 падает
echo "шаг 3: готово, всё прошло успешно"
