# SOXL/SOXS trading bot container image.
# Build:  podman build -t soxs-bot .
# Run:    podman run --rm --env-file .env -v ./logs:/app/logs:Z soxs-bot
#
# Named "Containerfile" (Podman's native default). It is a standard OCI
# Dockerfile, so `docker build` works against it too.
#
# Python 3.11 (not 3.14) so 32-bit Raspberry Pi builds can pull prebuilt
# numpy/pandas wheels from piwheels — which only targets Raspberry Pi OS's
# Python (3.11 on Bookworm). On x86_64/aarch64 pip still finds normal wheels.
FROM python:3.11-slim

# Fail fast, stream logs, no .pyc clutter in the image. PIP_EXTRA_INDEX_URL adds
# piwheels as a fallback wheel source for 32-bit ARM (ignored on other arches,
# which already have PyPI wheels).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://www.piwheels.org/simple

WORKDIR /app

# piwheels' numpy wheels link against the system OpenBLAS rather than bundling
# it, so install the runtime lib (no-op cost on other arches).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached across code changes.
COPY requirements-pi.txt requirements-finbert.txt ./
RUN pip install --no-cache-dir -r requirements-pi.txt

# FinBERT extras (torch + transformers) for news sentiment. torch has no 32-bit
# ARM wheels, so on armv7l we skip them on purpose and the bot falls back to
# VADER. On every other arch (x86_64/aarch64) the install is required, so a
# failure fails the build loudly rather than silently disabling FinBERT.
RUN arch="$(uname -m)"; \
    if [ "$arch" = "armv7l" ] || [ "$arch" = "armv6l" ]; then \
        echo "32-bit ARM ($arch): skipping FinBERT extras (no torch wheels); VADER fallback"; \
    else \
        pip install --no-cache-dir -r requirements-finbert.txt; \
    fi

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
