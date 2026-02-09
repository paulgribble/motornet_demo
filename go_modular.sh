#!/bin/sh

if [ -z "$1" ]; then
  echo "Usage: $0 <name>"
  exit 1
fi

uv run python reaching_model.py create "$1" --modular
uv run python reaching_model.py train "$1" --batches 5000
uv run python reaching_model.py test "$1"
