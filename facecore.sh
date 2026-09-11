#!/bin/sh
# Shell entrypoint wrapping `python -m facecore.cli` (plan Task 8).
# Prefer python3 (setup-python provides both; bare systems may lack `python`).
if command -v python3 >/dev/null 2>&1; then
    exec python3 -m facecore.cli "$@"
else
    exec python -m facecore.cli "$@"
fi
