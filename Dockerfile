FROM python:3.14.7-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# The app only needs to write its database, so it runs unprivileged and owns
# only the data directory.
RUN groupadd --system --gid 10001 mdredd \
    && useradd --system --uid 10001 --gid 10001 --no-create-home mdredd \
    && mkdir -p /app/data \
    && chown mdredd:mdredd /app/data
USER mdredd

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
