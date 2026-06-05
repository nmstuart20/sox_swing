# SOXL/SOXS trading bot container image.
# Build:  podman build -t soxs-bot .
# Run:    podman run --rm --env-file .env -v ./logs:/app/logs:Z soxs-bot
#
# Named "Containerfile" (Podman's native default). It is a standard OCI
# Dockerfile, so `docker build` works against it too.
FROM python:3.14-slim

# Fail fast, stream logs, no .pyc clutter in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

# Run as an unprivileged user; pre-create the logs mount point and hand it over.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# The bot installs SIGINT/SIGTERM handlers and shuts down cleanly, so plain
# exec form is enough for `podman stop` to flatten positions / flush logs.
CMD ["python", "main.py"]
