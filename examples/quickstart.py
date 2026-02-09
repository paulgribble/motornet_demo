"""
Quickstart example for ReachingModel.

This script demonstrates how to use the ReachingModel API to:
1. Create a new model (4-module: PMd, M1, S1, SC)
2. Train it on reaching movements
3. Test it on center-out reaching
4. Save and reload the model
5. Use custom configuration parameters
"""

from reaching_model import ReachingModel
import numpy as np

# =============================================================================
# Example 1: Create and train a model with default settings
# =============================================================================

print("=" * 60)
print("Example 1: Default Model (PMd, M1, S1, SC)")
print("=" * 60)

# Create a new 4-module model with default settings
model = ReachingModel.create(name="example_default")

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
# Example 2: Create a model with custom module sizes
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Custom Module Sizes")
print("=" * 60)

# Create a model with larger PMd and M1, smaller S1 and SC
model = ReachingModel.create(
    name="example_custom_sizes",
    module_sizes=[512, 256, 128, 64]
)

# Train the model
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
model = ReachingModel.load("example_default")
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

model = ReachingModel.load("example_default")

# Test with a velocity-dependent curl force field
model.test(
    n_targets=8,
    ff_strength=15.0  # Force field strength
)


# =============================================================================
# Example 5: Custom sensory delays and noise
# =============================================================================

print("\n" + "=" * 60)
print("Example 5: Custom Sensory Configuration")
print("=" * 60)

# Create a model with custom sensory delays and noise levels
# This simulates different neural processing conditions
model = ReachingModel.create(
    name="example_custom_sensory",

    # Episode settings
    episode_duration=2.5,           # Shorter episodes (2.5s instead of 3.0s)

    # Sensory delays (in seconds)
    proprioception_delay=0.03,      # Slower proprioceptive feedback (30ms vs 20ms)
    vision_delay=0.10,              # Slower visual feedback (100ms vs 80ms)

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
# Example 6: Custom connectivity masks
# =============================================================================

print("\n" + "=" * 60)
print("Example 6: Custom Connectivity")
print("=" * 60)

# Create a model with custom connectivity patterns
# This allows you to design specific network architectures

# Define a custom connectivity matrix (4 modules: PMd, M1, S1, SC)
# Values represent connection probability between modules
# Row i, Col j = probability of connection FROM module i TO module j
custom_connectivity = [
    [0.80, 0.40, 0.05, 0.10],   # PMd: strong self + M1 projection
    [0.20, 0.80, 0.10, 0.35],   # M1: strong self + SC (corticospinal)
    [0.15, 0.30, 0.80, 0.08],   # S1: strong self + M1 (online correction)
    [0.05, 0.10, 0.20, 0.80]    # SC: strong self + ascending to S1
]

model = ReachingModel.create(
    name="example_custom_conn",

    # Module architecture
    module_sizes=[256, 256, 128, 64],

    # Input connectivity masks (which modules receive which inputs)
    # Values are connection probabilities for each module [PMd, M1, S1, SC]
    vision_mask=[0.40, 0.10, 0.00, 0.00],     # Vision mainly to PMd
    proprio_mask=[0.00, 0.10, 0.30, 0.50],    # Proprioception mainly to SC and S1
    task_mask=[0.40, 0.10, 0.00, 0.00],       # Task info mainly to PMd

    # Inter-module connectivity
    connectivity_mask=custom_connectivity,

    # Output connectivity (which modules drive muscles)
    output_mask=[0.00, 0.00, 0.00, 0.60],     # Output only from SC

    # Recurrent dynamics
    spectral_scaling=1.2,            # Slightly stronger recurrence

    # Sensory parameters
    episode_duration=3.0,
    proprioception_delay=0.02,
    vision_delay=0.08,
    learning_rate=1e-3
)

print(f"Created model with custom connectivity:")
print(f"  Module sizes: {model.config.module_sizes}")
print(f"  Vision mask: {model.config.vision_mask}")
print(f"  Proprio mask: {model.config.proprio_mask}")
print(f"  Output mask: {model.config.output_mask}")
print(f"  Connectivity matrix:")
for i, row in enumerate(model.config.connectivity_mask):
    print(f"    Module {i} ({model.config.module_names[i]}): {row}")

# Train the custom model
model.train(n_batches=100, batch_size=32, plot_interval=0)
model.test(n_targets=8)


# =============================================================================
# Example 7: High-noise / challenging conditions
# =============================================================================

print("\n" + "=" * 60)
print("Example 7: Challenging Sensorimotor Conditions")
print("=" * 60)

# Simulate challenging conditions (e.g., aging, neurological conditions)
model = ReachingModel.create(
    name="example_challenging",

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
print("  - example_default/")
print("  - example_custom_sizes/")
print("  - example_custom_sensory/")
print("  - example_custom_conn/")
print("  - example_challenging/")


# =============================================================================
# Cleanup (optional)
# =============================================================================

# Uncomment these lines to delete the example models:
# import shutil
# for name in ["example_default", "example_custom_sizes", "example_custom_sensory",
#              "example_custom_conn", "example_challenging"]:
#     shutil.rmtree(name, ignore_errors=True)
