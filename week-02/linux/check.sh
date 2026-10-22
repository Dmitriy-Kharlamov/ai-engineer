#!/usr/bin/env bash
set -euo pipefail

# Шаг 1: ровно один аргумент
if [ "$#" -ne 1 ]; then
    echo "использование: $0 <файл>" >&2
    exit 1
fi

# Шаг 2: файл существует
file="$1"
if [ ! -f "$file" ]; then
    echo "ошибка: файл не найден: $file" >&2
    exit 1
fi

# Шаг 3: число строк (wc на macOS дополняет число пробелами — убираем их)
lines=$(wc -l < "$file")
echo "строк в $file: ${lines// /}"
