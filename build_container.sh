#!/usr/bin/env bash
#
# Build the SOXL/SOXS bot container image.
#
# The build downloads the FinBERT weights from huggingface.co (baked into the
# image so the bot runs offline — see Containerfile). Some networks' DNS filters
# the huggingface.co domain family (huggingface.co, cas-bridge.xethub.hf.co,
# ...), which makes the download fail with "Name or service not known". We pass
# public resolvers with --dns so the build bypasses the local/router DNS; this
# is harmless on networks that don't filter HF.
#
# Usage:
#   ./build_container.sh                 # build, tag soxs-bot
#   ./build_container.sh -t myname:dev   # any extra args pass through to podman
#   IMAGE=foo ENGINE=docker ./build_container.sh
#
set -euo pipefail

# Container engine and image tag are overridable via env.
ENGINE="${ENGINE:-podman}"
IMAGE="${IMAGE:-soxs-bot}"

# Public resolvers used only for this build, to dodge DNS that blocks HF.
DNS_FLAGS=(--dns=1.1.1.1 --dns=8.8.8.8)

# Run from the repo root (where the Containerfile lives) regardless of CWD.
cd "$(dirname "$0")"

echo "Building ${IMAGE} with ${ENGINE} (DNS: 1.1.1.1, 8.8.8.8)..."
exec "${ENGINE}" build "${DNS_FLAGS[@]}" -t "${IMAGE}" "$@" .
