#!/bin/bash
#
# Train 10 modular reaching models in parallel with live tqdm progress bars.
# Usage: ./go_modular_parallel.sh
#
# Ctrl-C cleanly kills all jobs.

N=10

trap 'kill 0; exit 1' INT

# Phase 1: Create all models (fast, do it sequentially to keep output clean)
for i in $(seq 1 $N); do
  NAME=$(printf "demo_modular_%02d" $i)
  uv run python reaching_model.py create "$NAME" &
done

wait

# precompile
uv run python reaching_model.py train demo_modular_01 --batches 1

# Phase 2: Train in parallel with tqdm position bars
#printf '\n%.0s' $(seq 1 $N)

for i in $(seq 1 $N); do
  NAME=$(printf "demo_modular_%02d" $i)
  POS=$((i - 1))
  (
    uv run python reaching_model.py train "$NAME" --batches 20000 --batch-size 32 --position $POS 2>&2 >>"${NAME}.log"
    uv run python reaching_model.py train "$NAME" --batches 1000 --task center_out --position $POS 2>&2 >>"${NAME}.log"
  ) &
done

wait

# Phase 3: Test all models (fast, sequential is fine)
echo ""
for i in $(seq 1 $N); do
  NAME=$(printf "demo_modular_%02d" $i)
  uv run python reaching_model.py test "$NAME" &
done

wait

echo "All done"


