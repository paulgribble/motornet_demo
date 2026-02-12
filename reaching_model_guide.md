# Reaching Model Guide

This document explains how `reaching_model.py` works and what you can do with it.

## Overview

`reaching_model.py` provides a high-level interface for training a modular recurrent neural network (GRU) to control a simulated biomechanical arm performing reaching movements. It wraps the [MotorNet](https://github.com/motornet-org/MotorNet) library and handles model creation, training, testing, and saving through both a Python API and a command-line interface.

The simulated arm is a 2-joint (shoulder + elbow), 6-muscle model (`RigidTendonArm26`). The neural network receives sensory feedback (vision, proprioception) and task instructions (target position, go cue), and must learn to produce muscle activation patterns that move the hand to the target.

## Architecture

The architecture is a 4-module modular GRU inspired by the primate motor system:

```
Task inputs ──→ [PMd] ──→ [M1] ──→ [SC] ──→ sigmoid ──→ muscle activations ──→ arm model
(target, go)               ↑↓        ↑       (1-step     (6 muscles)            (2 joints)
Vision ──────→ [PMd],[M1] [S1] ←─── [SC]      delay)
(hand pos)                  │
Proprioception ──────────→ [SC]
(muscle state)
```

| Module | Name                          | Size | Role                                                    |
| ------ | ----------------------------- | ---- | ------------------------------------------------------- |
| 0      | Dorsal premotor cortex (PMd)  | 128  | Motor planning, receives task goals + vision             |
| 1      | Primary motor cortex (M1)     | 128  | Motor command generation, receives vision + PMd commands |
| 2      | Somatosensory cortex (S1)     | 128  | Processes ascending proprioceptive signals from SC       |
| 3      | Spinal cord (SC)              | 16   | Alpha motor neurons, receives proprioception + M1 input  |

The connectivity between modules reflects known primate neuroanatomy. The `connectivity_mask` is indexed as **[receiver, sender]** — row *i* specifies the probability that module *i* receives connections from each sender module (columns).

Key inter-module pathways:
- **PMd ↔ M1** (0.20): Bidirectional premotor-motor connectivity.
- **M1 ↔ S1** (0.20): Bidirectional motor-somatosensory connectivity.
- **M1 → SC** (0.20): Corticospinal tract. Primary descending voluntary pathway.
- **SC → S1** (1.00): Ascending dorsal columns. Strong proprioceptive relay.

Sensory routing:
- Vision reaches PMd and M1 (dorsal stream).
- Task inputs (target, go cue) reach PMd only.
- Proprioception reaches SC only (direct muscle afferents).
- Only SC drives muscle output (motor neurons in the ventral horn), with a **1-timestep output delay**.

The activation function is `rect_tanh` (rectified tanh): `max(0, tanh(x))`.

## Python API

### Creating a Model

```python
from reaching_model import ReachingModel

# Create a model with default settings
model = ReachingModel.create("my_model")

# Create with custom module sizes
model = ReachingModel.create("big_model", module_sizes=[256, 256, 128, 32])
```

All configurable parameters at creation time:

|       Parameter        |       Default        |               Description               |
| ---------------------- | -------------------- | --------------------------------------- |
| `module_sizes`         | [128, 128, 128, 16]  | Units per module (PMd, M1, S1, SC)      |
| `episode_duration`     | 3.0                  | Simulation duration in seconds          |
| `proprioception_delay` | 0.01                 | Proprioceptive feedback delay (seconds) |
| `vision_delay`         | 0.11                 | Visual feedback delay (seconds)         |
| `proprioception_noise` | 1e-3                 | Proprioceptive noise (std dev)          |
| `vision_noise`         | 1e-3                 | Visual noise (std dev)                  |
| `action_noise`         | 1e-4                 | Motor command noise (std dev)           |
| `learning_rate`        | 1e-3                 | Adam optimizer learning rate            |
| `activation`           | "rect_tanh"          | Activation function (tanh or rect_tanh) |
| `output_delay`         | 1                    | Timesteps of delay on output layer      |

Connectivity parameters (advanced):

|      Parameter      |                       Default                        |                          Description                          |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| `vision_mask`       | [1.00, 1.00, 0.00, 0.00]                             | Connection probability from vision to each module             |
| `proprio_mask`      | [0.00, 0.00, 0.00, 1.00]                             | Connection probability from proprioception to each module     |
| `task_mask`         | [1.00, 0.00, 0.00, 0.00]                             | Connection probability from task inputs to each module        |
| `connectivity_mask` | 4x4 matrix (indexed as [receiver, sender])            | Inter-module connection probabilities                         |
| `output_mask`       | [0.00, 0.00, 0.00, 1.00]                             | Connection probability from each module to output             |
| `module_names`      | ["premotor", "motor", "somatosensory", "spinal"]     | Names for each module                                         |
| `spectral_scaling`  | 1.30                                                  | Spectral radius scaling for recurrent weights                 |

### Training

```python
model.train(n_batches=10000, batch_size=64)

# Train with a velocity-dependent curl force field
model.train(n_batches=5000, ff_strength=15.0)

# Train on center-out reaches only
model.train(n_batches=10000, task_mode="center_out")
```

|    Parameter    | Default  |             Description              |
| --------------- | -------- | ------------------------------------ |
| `n_batches`     | 10000    | Number of training batches           |
| `batch_size`    | 64       | Trials per batch                     |
| `ff_strength`   | 0.0      | Curl force field strength (0 = none) |
| `task_mode`     | "random" | "random" or "center_out"             |
| `plot_interval` | 100      | Batches between plot saves (0 = off) |
| `quiet`         | False    | Suppress progress bar                |

Training automatically saves intermediate plots (loss curves, hand paths, neural/muscle signals) and saves the model when finished. Training can be resumed — calling `train()` again continues from where it left off.

### Testing

```python
results = model.test(n_targets=8)

# Test with a force field
results = model.test(n_targets=8, ff_strength=15.0)
```

|     Parameter     |     Default      |           Description           |
| ----------------- | ---------------- | ------------------------------- |
| `n_targets`       | 8                | Number of targets in a circle   |
| `ff_strength`     | 0.0              | Curl force field strength       |
| `simulation_time` | episode_duration | Override simulation duration    |
| `save_plots`      | True             | Save hand path and signal plots |
| `save_data`       | True             | Save episode data as pickle     |

Testing runs center-out reaches with fixed timing (target at 0.5s, go cue at 1.0s, no catch trials) for reproducible evaluation.

### Saving and Loading

```python
model.save()                        # Save to current name
model.save("my_model_backup")       # Save with a new name

model = ReachingModel.load("my_model")  # Load from disk
print(model.summary())                  # Print model info
```

Each model is stored as a directory containing:
- `{name}_config.json` — all configuration parameters
- `{name}_weights.pkl` — network weights (PyTorch state dict)
- `{name}_training_state.json` — training progress and loss history

## Command-Line Interface

All operations are available from the command line:

```bash
# Create a model (uses default parameters)
uv run reaching_model.py create my_model

# Train
uv run reaching_model.py train my_model --batches 10000
uv run reaching_model.py train my_model --batches 5000 --ff 15.0

# Test
uv run reaching_model.py test my_model --targets 8
uv run reaching_model.py test my_model --targets 8 --ff 15.0

# Show model info
uv run reaching_model.py info my_model

# Save with a new name
uv run reaching_model.py save my_model my_model_backup

# Show architecture details
uv run reaching_model.py arch
```

CLI options for `train`: `--batches`, `--batch-size`, `--ff`, `--task` (random or center_out). CLI options for `test`: `--targets`, `--ff`. To customize model parameters (delays, noise, module sizes, etc.), use the Python API.

## Task Structure

Each trial has three phases:

```
|--- Hold ---|--- Target visible ---|--- Go cue (reach) ---|
0s        ~0.3-0.8s              ~0.8-1.3s              3.0s
```

1. **Hold period**: The network sees the start position as "target". No go cue.
2. **Target cue**: The target switches to the final reach position. Go cue is still off. The network should plan but not move.
3. **Go cue**: The go signal turns on. The network should move the hand to the target.

During training, 50% of trials are **catch trials** where the go cue never appears — the network must learn to hold still even after seeing the target.

### Task Inputs to the Network

The network receives at each timestep:
- **Task inputs** (3 values): target x, target y, go cue (0 or 1)
- **Vision** (2 values): noisy fingertip x, y (delayed by `vision_delay`)
- **Proprioception** (12 values): normalized muscle lengths and velocities (delayed by `proprioception_delay`)

## Loss Function

The model uses `michaels_modular_loss`, which penalizes four terms:

|    Component     | Weight |                   What it penalizes                    |
| ---------------- | ------ | ------------------------------------------------------ |
| position         | 1e+3   | L1 hand-to-target distance (summed over x, y)         |
| muscle           | 1e-1   | Muscle force magnitude (summed over muscles)           |
| hidden_derivative| 1e+3   | 2nd derivative of hidden activity (spectral smoothness)|
| jerk             | 1e+5   | 2nd derivative of velocity (movement jerk)             |

## Force Fields

A velocity-dependent curl force field can be applied during training and/or testing. This pushes the hand perpendicular to its movement direction, forcing the network to learn compensatory muscle patterns.

```python
# Train in a force field
model.train(n_batches=5000, ff_strength=15.0)

# Test adaptation
model.test(n_targets=8, ff_strength=15.0)
```

The force field is only active after the go cue (not during the hold period).

## Outputs and Plots

### Training Outputs

During training, the model directory accumulates:
- `{name}_losses.png` — loss curves (linear and log scale)
- `{name}_handpaths.png` — hand trajectories from the latest batch
- `{name}_signals_{0-3}.png` — detailed time series for 4 sample trials

### Test Outputs

After testing:
- `{name}_test_handpaths.png` (or `_ff15.png` with force field) — all reach trajectories
- `{name}_test_trial_{i}.png` — per-trial breakdown for each target
- `{name}_test_data.pkl` — raw episode data (numpy arrays)

### Signal Plots

Each signal plot shows 6 panels for a single trial:
1. **Go cue** — binary signal
2. **X position** — input (dotted), actual (solid), target (dashed)
3. **Y position** — same format
4. **Velocity** — hand speed over time
5. **GRU hidden activity** — all hidden units
6. **Muscle activations** — all 6 muscles (0 to 1)

## Typical Workflows

### Train a model from scratch

```python
model = ReachingModel.create("baseline")
model.train(n_batches=10000)
model.test(n_targets=8)
```

### Study force field adaptation

```python
# Train without force field
model = ReachingModel.create("ff_study")
model.train(n_batches=10000)
model.test(n_targets=8)                    # Baseline performance

# Continue training with force field
model.train(n_batches=5000, ff_strength=15.0)
model.test(n_targets=8, ff_strength=15.0)  # Adapted performance
model.test(n_targets=8, ff_strength=0.0)   # After-effects
```

### Vary sensory parameters (Python API only)

```python
# High noise, long delays
noisy = ReachingModel.create("noisy",
    proprioception_noise=0.01,
    vision_noise=0.01,
    proprioception_delay=0.05,
    vision_delay=0.15
)
noisy.train(n_batches=10000)

# No vision delay
fast_vision = ReachingModel.create("fast_vision", vision_delay=0.0)
fast_vision.train(n_batches=10000)
```

### Access episode data programmatically

```python
model = ReachingModel.load("my_model")
results = model.test(n_targets=8, save_plots=False, save_data=False)

xy = results['xy']                       # (8, timesteps, 4) — x, y, vx, vy
hidden = results['hidden']               # (8, timesteps, hidden_dim)
muscle = results['muscle']               # (8, timesteps, 6)
targets = results['targets']             # (8, timesteps, 2)
actions = results['actions']             # (8, timesteps, 6)
joint = results['joint']                 # (8, timesteps, 4) — shoulder, elbow, vel_s, vel_e
delay_tg_times = results['delay_tg_times']  # (8,) — timestep when target appears
delay_go_times = results['delay_go_times']  # (8,) — timestep when go cue appears
```
