#!/bin/bash
# Command-line examples for ReachingModel
# Run these from the motornet_demo directory

# =============================================================================
# Basic Model Creation
# =============================================================================

# Create a simple GRU model with 256 units (default)
uv run python reaching_model.py create my_model

# Create a simple GRU model with custom unit count
uv run python reaching_model.py create small_model --units 64

# Create a modular model with default architecture (256, 256, 64)
uv run python reaching_model.py create my_modular --modular

# Create a modular model with custom module sizes
uv run python reaching_model.py create custom_modular --modular --module-sizes 128 128 32


# =============================================================================
# Custom Configuration - Simple Models
# =============================================================================

# Create a model with slower sensory delays (e.g., simulating aging)
uv run python reaching_model.py create slow_feedback \
    --units 256 \
    --proprio-delay 0.04 \
    --vision-delay 0.12

# Create a model with high sensory noise (challenging conditions)
uv run python reaching_model.py create noisy_model \
    --units 256 \
    --proprio-noise 0.005 \
    --vision-noise 0.003 \
    --action-noise 0.0005

# Create a model with shorter episodes and slower learning
uv run python reaching_model.py create short_episodes \
    --units 256 \
    --duration 2.0 \
    --learning-rate 0.0005

# Create a model with all custom parameters
uv run python reaching_model.py create fully_custom \
    --units 512 \
    --duration 2.5 \
    --proprio-delay 0.03 \
    --vision-delay 0.10 \
    --proprio-noise 0.002 \
    --vision-noise 0.001 \
    --action-noise 0.0001 \
    --learning-rate 0.0008


# =============================================================================
# Custom Configuration - Modular Models
# =============================================================================

# Create a modular model with custom connectivity patterns
# Vision mainly to module 0, proprioception mainly to module 2, output from module 2
uv run python reaching_model.py create custom_connectivity \
    --modular \
    --module-sizes 200 100 50 \
    --vision-mask 0.3 0.1 0.0 \
    --proprio-mask 0.0 0.1 0.6 \
    --task-mask 0.3 0.1 0.0 \
    --output-mask 0.0 0.1 0.6 \
    --spectral-scaling 1.2

# Create a hierarchical "cortical" model
# Premotor -> Motor -> Spinal hierarchy
uv run python reaching_model.py create cortical_hierarchy \
    --modular \
    --module-sizes 256 256 64 \
    --vision-mask 0.3 0.05 0.0 \
    --proprio-mask 0.0 0.05 0.5 \
    --task-mask 0.3 0.05 0.0 \
    --output-mask 0.0 0.0 0.6 \
    --spectral-scaling 1.1

# Modular model with custom delays and noise
uv run python reaching_model.py create modular_custom_all \
    --modular \
    --module-sizes 128 128 32 \
    --proprio-delay 0.025 \
    --vision-delay 0.08 \
    --proprio-noise 0.002 \
    --vision-noise 0.001 \
    --learning-rate 0.0008


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


# =============================================================================
# Typical Workflow Examples
# =============================================================================

# --- Workflow 1: Basic experiment ---
uv run python reaching_model.py create experiment1 --units 256
uv run python reaching_model.py train experiment1 --batches 10000
uv run python reaching_model.py test experiment1 --targets 8

# --- Workflow 2: Force field adaptation ---
uv run python reaching_model.py create ff_experiment --units 256
uv run python reaching_model.py train ff_experiment --batches 10000
uv run python reaching_model.py test ff_experiment --targets 8           # Baseline
uv run python reaching_model.py train ff_experiment --batches 5000 --ff 15.0
uv run python reaching_model.py test ff_experiment --targets 8 --ff 15.0 # After adaptation

# --- Workflow 3: Comparing architectures ---
uv run python reaching_model.py create simple_256 --units 256
uv run python reaching_model.py create simple_512 --units 512
uv run python reaching_model.py create modular_std --modular
uv run python reaching_model.py train simple_256 --batches 10000
uv run python reaching_model.py train simple_512 --batches 10000
uv run python reaching_model.py train modular_std --batches 10000
uv run python reaching_model.py test simple_256
uv run python reaching_model.py test simple_512
uv run python reaching_model.py test modular_std

# --- Workflow 4: Sensory perturbation study ---
# Normal conditions
uv run python reaching_model.py create normal_sensory --units 256
uv run python reaching_model.py train normal_sensory --batches 10000
uv run python reaching_model.py test normal_sensory

# Delayed visual feedback
uv run python reaching_model.py create delayed_vision --units 256 --vision-delay 0.15
uv run python reaching_model.py train delayed_vision --batches 10000
uv run python reaching_model.py test delayed_vision

# Noisy proprioception
uv run python reaching_model.py create noisy_proprio --units 256 --proprio-noise 0.01
uv run python reaching_model.py train noisy_proprio --batches 10000
uv run python reaching_model.py test noisy_proprio


# =============================================================================
# Parameter Reference
# =============================================================================

# Show all available options for create command:
uv run python reaching_model.py create --help

# Show all available options for train command:
uv run python reaching_model.py train --help

# Show all available options for test command:
uv run python reaching_model.py test --help
