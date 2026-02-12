#!/bin/sh

if [ -z "$1" ]; then
  echo "Usage: $0 <name> [position]"
  exit 1
fi

POS_ARG=""
if [ -n "$2" ]; then
  POS_ARG="--position $2"
fi

uv run python reaching_model.py create "$1"
uv run python reaching_model.py train "$1" --batches 1 $POS_ARG
uv run python reaching_model.py train "$1" --batches 20000 --batch-size 32 $POS_ARG
uv run python reaching_model.py train "$1" --batches 1000 --task center_out $POS_ARG
uv run python reaching_model.py test "$1"
