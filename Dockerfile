# syntax=docker/dockerfile:1
# ragkit API. Этапы:
#   deps    — venv с зависимостями прода и установленным пакетом (не editable)
#   test    — deps + зависимости разработки + tests/:  docker build --target test -t ragkit:test .
#   runtime — только venv из deps, без pip-кешей, компиляторов и исходников; по умолчанию
#
# Сборка и запуск:
#   docker build -t ragkit .
#   docker run --rm -p 8000:8000 -e APP_API_KEY=... -e APP_JWT_SECRET=... ragkit
# Секретов в образе нет: они приходят переменными окружения при запуске (Ф3-08, Ф3-11).

ARG PYTHON_IMAGE=python:3.13-slim

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- deps: от редко меняющегося к часто меняющемуся ------------------------------------
FROM base AS deps
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
# 1) зависимости: слой пересобирается только при правке requirements-prod.txt
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt
# 2) сам пакет: правка кода пересобирает только этот слой (секунды, а не минуты)
COPY pyproject.toml README.md ./
COPY ragkit/ ragkit/
COPY api/ api/
COPY services/ services/
COPY tools/ tools/
# не editable: метаданные пакета лежат в venv, importlib.metadata.version("ragkit") работает
RUN pip install --no-deps .

# --- test: всё для pytest, в прод не попадает -----------------------------------------
FROM deps AS test
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY tests/ tests/
CMD ["python", "-m", "pytest", "-p", "no:cacheprovider"]

# --- runtime --------------------------------------------------------------------------
FROM base AS runtime
COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# не root: уязвимость в сервисе не даёт root внутри контейнера
RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /home/app
EXPOSE 8000
# готовность, а не живость: /health/ready проверяет, что эмбеддер выдаёт векторы
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]
# exec-форма: PID 1 — сам uvicorn, SIGTERM от docker stop доходит до него
ENTRYPOINT ["uvicorn", "api.main:create_app", "--factory"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
