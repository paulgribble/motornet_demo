#!/bin/sh

if [ -z "$1" ]; then
  echo "Usage: $0 <name>"
  exit 1
fi

uv run python reaching_model.py create "$1"
uv run python reaching_model.py train "$1" --batches 10000 --batch-size 32
uv run python reaching_model.py train "$1" --batches 500 --task center_out
uv run python reaching_model.py test "$1"
