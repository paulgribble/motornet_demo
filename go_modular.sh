#!/bin/sh

uv run python reaching_model.py create demo1_modular --modular
uv run python reaching_model.py train demo1_modular --batches 2000
uv run python reaching_model.py test demo1_modular
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 demo1_modular_visualize.ipynb
