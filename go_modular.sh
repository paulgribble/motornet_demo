#!/bin/sh

uv run python reaching_model.py create "$1" --modular
uv run python reaching_model.py train "$1" --batches 20000
uv run python reaching_model.py test "$1"
