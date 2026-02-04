"""
Quickstart example for ReachingModel.

This script demonstrates how to use the ReachingModel API to:
1. Create a new model
2. Train it on reaching movements
3. Test it on center-out reaching
4. Save and reload the model
5. Use custom configuration parameters
"""

from reaching_model import ReachingModel
import numpy as np

# =============================================================================
# Example 1: Create and train a simple model
# =============================================================================

print("=" * 60)
print("Example 1: Simple GRU Model")
print("=" * 60)

# Create a new model with 128 hidden units
model = ReachingModel.create(
    name="example_simple",
    n_units=128,
    modular=False
)

# Train for a small number of batches (for demonstration)
# In practice, you'd use 10000+ batches
model.train(
    n_batches=100,       # Number of training batches
    batch_size=32,       # Trials per batch
    ff_strength=0.0,     # No force field
    task_mode="random",  # Random reaches across workspace
    plot_interval=50     # Save plots every 50 batches
)

# Test on center-out reaching
model.test(
    n_targets=8,         # 8 targets in a circle
    ff_strength=0.0      # No force field
)

print(f"\nModel summary:\n{model}")


# =============================================================================
# Example 2: Create and train a modular model
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Modular GRU Model")
print("=" * 60)

# Create a modular model with 3 interconnected modules
model = ReachingModel.create(
    name="example_modular",
    modular=True,
    module_sizes=[128, 128, 32]  # Sizes of each module
)

# Train the modular model
model.train(
    n_batches=100,
    batch_size=32,
    plot_interval=50
)

# Test on center-out reaching
model.test(n_targets=8)

print(f"\nModel summary:\n{model}")


# =============================================================================
# Example 3: Load an existing model and continue training
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Load and Continue Training")
print("=" * 60)

# Load the previously saved model
model = ReachingModel.load("example_simple")
print(f"Loaded model with {model.training_state.batches_completed} batches trained")

# Continue training
model.train(n_batches=50, batch_size=32, plot_interval=0)
print(f"Now trained for {model.training_state.batches_completed} batches total")


# =============================================================================
# Example 4: Test with force field perturbation
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Test with Force Field")
print("=" * 60)

model = ReachingModel.load("example_simple")

# Test with a velocity-dependent curl force field
model.test(
    n_targets=8,
    ff_strength=15.0  # Force field strength
)


# =============================================================================
# Example 5: Custom configuration - Simple model with modified delays/noise
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Custom Configuration (Simple Model)")
print("=" * 60)

# Create a model with custom sensory delays and noise levels
# This simulates different neural processing conditions
model = ReachingModel.create(
    name="example_custom_simple",
    n_units=256,
    modular=False,

    # Episode settings
    episode_duration=2.5,           # Shorter episodes (2.5s instead of 3.0s)

    # Sensory delays (in seconds)
    proprioception_delay=0.03,      # Slower proprioceptive feedback (30ms vs 20ms)
    vision_delay=0.10,              # Slower visual feedback (100ms vs 70ms)

    # Noise levels (standard deviation)
    proprioception_noise=2e-3,      # More proprioceptive noise
    vision_noise=5e-4,              # Less visual noise
    action_noise=1e-5,              # Less motor noise

    # Learning parameters
    learning_rate=5e-4              # Slower learning rate
)

print(f"Created model with custom config:")
print(f"  Proprio delay: {model.config.proprioception_delay}s")
print(f"  Vision delay: {model.config.vision_delay}s")
print(f"  Proprio noise: {model.config.proprioception_noise}")
print(f"  Vision noise: {model.config.vision_noise}")
print(f"  Learning rate: {model.config.learning_rate}")

# Train with these settings
model.train(n_batches=100, batch_size=32, plot_interval=0)


# =============================================================================
# Example 6: Custom configuration - Modular model with modified connectivity
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: Custom Configuration (Modular Model)")
print("=" * 60)

# Create a modular model with custom connectivity patterns
# This allows you to design specific network architectures

# Define a custom connectivity matrix (3 modules)
# Values represent connection probability between modules
# Row i, Col j = probability of connection FROM module j TO module i
custom_connectivity = [
    [1.0, 0.1, 0.0],   # Module 0: receives from self (100%), module 1 (10%), not module 2
    [0.3, 1.0, 0.3],   # Module 1: receives from all (hub module)
    [0.0, 0.2, 1.0]    # Module 2: receives from module 1 (20%) and self (100%)
]

model = ReachingModel.create(
    name="example_custom_modular",
    modular=True,

    # Module architecture
    module_sizes=[200, 100, 50],     # Asymmetric module sizes

    # Input connectivity masks (which modules receive which inputs)
    # Values are connection probabilities for each module
    vision_mask=[0.3, 0.1, 0.0],     # Vision mainly to module 0
    proprio_mask=[0.0, 0.1, 0.6],    # Proprioception mainly to module 2
    task_mask=[0.3, 0.1, 0.0],       # Task info mainly to module 0

    # Inter-module connectivity
    connectivity_mask=custom_connectivity,

    # Output connectivity (which modules drive muscles)
    output_mask=[0.0, 0.1, 0.6],     # Output mainly from module 2

    # Recurrent dynamics
    spectral_scaling=1.2,            # Slightly stronger recurrence

    # Sensory parameters
    episode_duration=3.0,
    proprioception_delay=0.02,
    vision_delay=0.07,
    learning_rate=1e-3
)

print(f"Created modular model with custom connectivity:")
print(f"  Module sizes: {model.config.module_sizes}")
print(f"  Vision mask: {model.config.vision_mask}")
print(f"  Proprio mask: {model.config.proprio_mask}")
print(f"  Output mask: {model.config.output_mask}")
print(f"  Connectivity matrix:")
for i, row in enumerate(model.config.connectivity_mask):
    print(f"    Module {i} receives: {row}")

# Train the custom modular model
model.train(n_batches=100, batch_size=32, plot_interval=0)
model.test(n_targets=8)


# =============================================================================
# Example 7: Biologically-inspired configuration
# =============================================================================

print("\n" + "=" * 60)
print("Example 7: Biologically-Inspired Architecture")
print("=" * 60)

# Create a model inspired by cortical motor control hierarchy:
# - Module 0: "Premotor" - receives visual/task info, plans movements
# - Module 1: "Motor cortex" - integrates all info, central hub
# - Module 2: "Spinal/output" - receives proprioception, generates commands

model = ReachingModel.create(
    name="example_bio_inspired",
    modular=True,

    # Cortical hierarchy sizes
    module_sizes=[256, 256, 64],     # Premotor, Motor, Spinal

    # Input routing (mimics sensory pathways)
    vision_mask=[0.3, 0.05, 0.0],    # Vision -> Premotor (dorsal stream)
    task_mask=[0.3, 0.05, 0.0],      # Goals -> Premotor (prefrontal input)
    proprio_mask=[0.0, 0.05, 0.5],   # Proprioception -> Spinal (reflex arcs)

    # Hierarchical connectivity (feedforward + feedback)
    connectivity_mask=[
        [1.0, 0.2, 0.0],   # Premotor: self + feedback from Motor
        [0.3, 1.0, 0.2],   # Motor: feedforward from Premotor + feedback from Spinal
        [0.0, 0.3, 1.0]    # Spinal: feedforward from Motor + self
    ],

    # Output only from "spinal" module
    output_mask=[0.0, 0.0, 0.6],

    # Realistic delays
    proprioception_delay=0.02,       # Fast spinal reflexes
    vision_delay=0.08,               # Slower visual processing

    # Moderate noise
    proprioception_noise=1e-3,
    vision_noise=1e-3,
    action_noise=1e-4,

    spectral_scaling=1.1
)

print("Created biologically-inspired hierarchical model")
print("  Premotor (256 units): receives vision + task goals")
print("  Motor cortex (256 units): integration hub")
print("  Spinal (64 units): proprioception + motor output")

model.train(n_batches=100, batch_size=32, plot_interval=0)


# =============================================================================
# Example 8: High-noise / challenging conditions
# =============================================================================

print("\n" + "=" * 60)
print("Example 8: Challenging Sensorimotor Conditions")
print("=" * 60)

# Simulate challenging conditions (e.g., aging, neurological conditions)
model = ReachingModel.create(
    name="example_challenging",
    n_units=256,
    modular=False,

    # Increased delays (slower processing)
    proprioception_delay=0.05,       # 50ms proprioceptive delay
    vision_delay=0.15,               # 150ms visual delay

    # Increased noise (less reliable signals)
    proprioception_noise=5e-3,       # 5x normal proprio noise
    vision_noise=3e-3,               # 3x normal visual noise
    action_noise=5e-4,               # 5x normal motor noise

    # Longer episodes to allow for slower movements
    episode_duration=4.0,

    # Slower learning (may need more training)
    learning_rate=5e-4
)

print("Created model with challenging sensorimotor conditions:")
print(f"  Proprioception delay: {model.config.proprioception_delay * 1000:.0f}ms")
print(f"  Vision delay: {model.config.vision_delay * 1000:.0f}ms")
print(f"  High noise levels simulating degraded sensory signals")

model.train(n_batches=100, batch_size=32, plot_interval=0)


print("\n" + "=" * 60)
print("All examples complete!")
print("=" * 60)
print("\nCheck these directories for results:")
print("  - example_simple/")
print("  - example_modular/")
print("  - example_custom_simple/")
print("  - example_custom_modular/")
print("  - example_bio_inspired/")
print("  - example_challenging/")


# =============================================================================
# Cleanup (optional)
# =============================================================================

# Uncomment these lines to delete the example models:
# import shutil
# for name in ["example_simple", "example_modular", "example_custom_simple",
#              "example_custom_modular", "example_bio_inspired", "example_challenging"]:
#     shutil.rmtree(name, ignore_errors=True)
