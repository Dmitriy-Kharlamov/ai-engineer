#!/usr/bin/env bash
set -euo pipefail

echo "шаг 1: начинаю"
cat /no/such/file.txt          # шаг 2 падает
echo "шаг 3: готово, всё прошло успешно"
