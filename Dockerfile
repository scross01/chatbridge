FROM python:3.13-slim

LABEL description="OpenAI API compatible endpoint for the OCI Generative AI Inference APIs"

ENV USER=chatbridge \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash $USER

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
ENV APP_DIR=/app
ADD chatbridge $APP_DIR/chatbridge
COPY pyproject.toml uv.lock README.md $APP_DIR/
WORKDIR $APP_DIR

# Sync the project into a new environment
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PYTHONPATH=$APP_DIR
RUN chown -R "$USER":"$USER" $APP_DIR
USER $USER

ENV OCI_CONFIG_PATH=~/.oci

EXPOSE 8000/tcp

CMD ["uvicorn", "chatbridge.main:app", "--host", "0.0.0.0"]
