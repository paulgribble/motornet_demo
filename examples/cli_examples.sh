#!/bin/bash
# Command-line examples for ReachingModel (4-module: PMd, M1, S1, SC)
# Run these from the motornet_demo directory

# =============================================================================
# Model Creation
# =============================================================================

# Create a model with default settings
uv run python reaching_model.py create my_model

# Create a model with custom sensory delays
uv run python reaching_model.py create slow_feedback \
    --proprio-delay 0.04 \
    --vision-delay 0.12

# Create a model with high sensory noise
uv run python reaching_model.py create noisy_model \
    --proprio-noise 0.005 \
    --vision-noise 0.003 \
    --action-noise 0.0005

# Create a model with shorter episodes and slower learning
uv run python reaching_model.py create short_episodes \
    --duration 2.0 \
    --learning-rate 0.0005


# =============================================================================
# Training Models
# =============================================================================

# Train with default settings (10000 batches, batch_size=64)
uv run python reaching_model.py train my_model

# Train with custom batch settings
uv run python reaching_model.py train my_model --batches 5000 --batch-size 32

# Train with a force field perturbation
uv run python reaching_model.py train my_model --batches 5000 --ff 15.0

# Train on center-out task instead of random reaches
uv run python reaching_model.py train my_model --batches 5000 --task center_out

# Train quietly (no progress bar)
uv run python reaching_model.py train my_model --batches 1000 --quiet

# Train without intermediate plots (faster)
uv run python reaching_model.py train my_model --batches 5000 --plot-interval 0


# =============================================================================
# Testing Models
# =============================================================================

# Test on center-out reaching with 8 targets (default)
uv run python reaching_model.py test my_model

# Test with different number of targets
uv run python reaching_model.py test my_model --targets 16

# Test with a force field
uv run python reaching_model.py test my_model --targets 8 --ff 15.0

# Test without saving plots (just data)
uv run python reaching_model.py test my_model --no-plots

# Test without saving data (just plots)
uv run python reaching_model.py test my_model --no-data


# =============================================================================
# Other Commands
# =============================================================================

# Show model information
uv run python reaching_model.py info my_model

# Save model with a new name (creates a copy)
uv run python reaching_model.py save my_model my_model_backup

# Show architecture details
uv run python reaching_model.py arch


# =============================================================================
# Typical Workflows
# =============================================================================

# --- Workflow 1: Basic experiment ---
uv run python reaching_model.py create experiment1
uv run python reaching_model.py train experiment1 --batches 10000
uv run python reaching_model.py test experiment1 --targets 8

# --- Workflow 2: Force field adaptation ---
uv run python reaching_model.py create ff_experiment
uv run python reaching_model.py train ff_experiment --batches 10000
uv run python reaching_model.py test ff_experiment --targets 8           # Baseline
uv run python reaching_model.py train ff_experiment --batches 5000 --ff 15.0
uv run python reaching_model.py test ff_experiment --targets 8 --ff 15.0 # After adaptation

# --- Workflow 3: Sensory perturbation study ---
uv run python reaching_model.py create normal_sensory
uv run python reaching_model.py train normal_sensory --batches 10000
uv run python reaching_model.py test normal_sensory

uv run python reaching_model.py create delayed_vision --vision-delay 0.15
uv run python reaching_model.py train delayed_vision --batches 10000
uv run python reaching_model.py test delayed_vision

uv run python reaching_model.py create noisy_proprio --proprio-noise 0.01
uv run python reaching_model.py train noisy_proprio --batches 10000
uv run python reaching_model.py test noisy_proprio


# =============================================================================
# Parameter Reference
# =============================================================================

# Show all available options:
uv run python reaching_model.py create --help
uv run python reaching_model.py train --help
uv run python reaching_model.py test --help
