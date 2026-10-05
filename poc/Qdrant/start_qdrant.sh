#!/usr/bin/env bash
# Starts the native Qdrant server used by main.py. No Docker involved:
# .qdrant-server/qdrant is the static musl build of v1.19.1, which matches
# the qdrant-client version pinned in the environment, so the client's
# compatibility check stays quiet.
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.qdrant-server"

# Storage lives next to the binary and is relative to the working directory,
# which is why we cd before exec.
export QDRANT__SERVICE__HOST=127.0.0.1
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334
export QDRANT__STORAGE__STORAGE_PATH=./storage
export QDRANT__STORAGE__SNAPSHOTS_PATH=./snapshots
# The benchmark drops and rebuilds both collections on every run, so telemetry
# phoning home would only add noise.
export QDRANT__TELEMETRY_DISABLED=true

cd "$SERVER_DIR"
exec ./qdrant "$@"
