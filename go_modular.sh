#!/bin/sh

if [ -z "$1" ]; then
  echo "Usage: $0 <name>"
  exit 1
fi

uv run python reaching_model.py create "$1"
<<<<<<< HEAD
uv run python reaching_model.py train "$1" --batches 10000 --batch-size 32
=======
uv run python reaching_model.py train "$1" --batches 20000 --batch-size 32
>>>>>>> fefb60ba57eab3243deb01c321a6adb0d70ec907
uv run python reaching_model.py train "$1" --batches 500 --task center_out
uv run python reaching_model.py test "$1"
