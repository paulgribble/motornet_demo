#!/bin/bash
set -e

# Ensure we have a .venv in the project
uv venv .venv

# Install everything defined in pyproject.toml / uv.lock
uv sync
