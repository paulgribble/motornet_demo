"""
Reaching Model - A simplified interface for training modular RNNs to control a biomechanical arm.

The architecture is a 4-module modular GRU inspired by the primate motor system:
  PMd (dorsal premotor) → M1 (motor cortex) → SC (spinal cord)
                          ↑ S1 (somatosensory cortex)

PMd holds the motor plan during the delay period; at the go cue, PMd→M1 transmits
initial conditions that launch execution dynamics. Output is from SC only, so cortical
delay activity is structurally output-null (Kaufman et al., 2014).

Example usage (API):
    from reaching_model import ReachingModel

    model = ReachingModel.create("my_model")
    model.train(n_batches=10000, batch_size=64)
    model.test(n_targets=8)
    model.save()

Example usage (CLI):
    uv run reaching_model.py create my_model
    uv run reaching_model.py train my_model --batches 10000
    uv run reaching_model.py test my_model --targets 8
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

# Set threading environment variables BEFORE importing torch/numpy
os.environ['OMP_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import torch as th
th.set_num_threads(1)
th.set_num_interop_threads(1)

import numpy as np
import matplotlib
if 'IPython' not in sys.modules:
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm

import motornet as mn
from my_env import ExperimentEnv
from my_task import ExperimentTask
from my_policy import ModularPolicyGRU
from my_loss import calculate_loss_mehrdad
from my_utils import run_episode, plot_losses, plot_handpaths, plot_signals


# =============================================================================
# 4-Module Architecture: PMd, M1, S1, SC
# =============================================================================
#
# Anatomically-motivated connectivity for a modular recurrent network
# controlling a 2-joint, 6-muscle arm model.
#
# Mask values are CONNECTION PROBABILITIES: during initialization, each
# potential synapse is included with this probability, creating structured
# sparsity that reflects known projection density in the primate motor system.
#
# Modules:
#   [0] PMd — Dorsal premotor cortex (256 units): receives target/go cue and
#             vision; computes motor plans during the delay period
#   [1] M1  — Primary motor cortex (256 units): receives plan from PMd,
#             generates temporally-patterned descending commands
#   [2] S1  — Somatosensory cortex (128 units): processes proprioception,
#             projects corrective signals to M1 and plan updates to PMd
#   [3] SC  — Spinal cord (64 units): alpha motor neurons + local interneuron
#             circuits; the only module that drives muscle output
#
# Key inter-module pathways:
#   PMd→M1  (0.35): Densest corticocortical motor projection. Primary pathway
#                    for converting plans into executable commands.
#   M1→SC   (0.30): Corticospinal tract. Primary descending voluntary pathway.
#   S1→M1   (0.25): Areas 3a/2 → M1. Critical for online proprioceptive
#                    correction during reaching.
#   SC→S1   (0.15): Ascending dorsal columns (cuneate → VPLc → S1).
#   M1→PMd  (0.15): Execution feedback / efference copy.
#   S1→PMd  (0.12): Proprioceptive plan updating (area 5 → PMd).
#   PMd→SC  (0.08): Weak direct PMd corticospinal projections.
#   SC→M1   (0.08): Long-loop transcortical reflex pathway.

MODULE_PRESET = dict(
    module_names=["premotor", "motor", "somatosensory", "spinal"],
    module_sizes=[256, 256, 128, 64],
    #                          PMd   M1    S1    SC
    vision_mask=              [0.50, 0.15, 0.00, 0.00],  # dorsal stream: V1→PPC→PMd (primary), PPC→M1 (weak)
    proprio_mask=             [0.00, 0.10, 0.40, 0.50],  # Ia/Ib: SC direct, S1 via dorsal cols, M1 via VPLo
    task_mask=                [0.50, 0.10, 0.00, 0.00],  # target/go: PMd primary (PFC/PPC), M1 weak
    connectivity_mask=[
        #                      →PMd  →M1   →S1   →SC
        # from PMd:            self  plan→  negl. weak CST
        #                            exec
        [                      0.70, 0.35, 0.02, 0.08],
        # from M1:             efference self  weak  CST
        #                      copy              fb
        [                      0.15, 0.70, 0.05, 0.30],
        # from S1:             plan  online self  weak
        #                      update corr.       CST
        [                      0.12, 0.25, 0.70, 0.05],
        # from SC:             negl. long  asc.  self
        #                            loop  dorsal (interneurons)
        #                                  cols
        [                      0.02, 0.08, 0.15, 0.70],
    ],
    output_mask=              [0.00, 0.00, 0.00, 0.50],  # alpha motor neurons in SC only
    spectral_scaling=1.15,  # slightly higher for richer preparatory dynamics
)


def print_architecture():
    """Print a readable summary of the 4-module architecture."""
    names = MODULE_PRESET['module_names']
    sizes = MODULE_PRESET['module_sizes']
    n_mod = len(names)
    conn = np.array(MODULE_PRESET['connectivity_mask'])

    print("=" * 70)
    print(f"4-MODULE ARCHITECTURE: {', '.join(n.upper() for n in names)}")
    print("=" * 70)

    print("\nModules:")
    for i, (name, size) in enumerate(zip(names, sizes)):
        print(f"  [{i}] {name:15s}  {size:4d} units")
    print(f"  {'Total':>20s}: {sum(sizes):4d} units")

    print("\nInput masks (connection probability):")
    header = "".join(f"{n:>8s}" for n in names)
    print(f"  {'':15s}{header}")
    for label, mask in [("Vision", MODULE_PRESET['vision_mask']),
                        ("Proprioception", MODULE_PRESET['proprio_mask']),
                        ("Task (tgt, go)", MODULE_PRESET['task_mask'])]:
        vals = "".join(f"{v:8.2f}" for v in mask)
        print(f"  {label:15s}{vals}")

    print(f"\nConnectivity (rows=from, cols=to):")
    header = "".join(f"{'→'+n:>8s}" for n in names)
    print(f"  {'':15s}{header}")
    for i, name in enumerate(names):
        row = conn[i]
        vals = "".join(f"{v:8.2f}" for v in row)
        marks = []
        for j in range(n_mod):
            if i == j:
                marks.append("self")
            elif row[j] >= 0.25:
                marks.append("★")
            elif row[j] >= 0.10:
                marks.append("●")
            elif row[j] >= 0.05:
                marks.append("○")
            else:
                marks.append("·")
        print(f"  {name:15s}{vals}   {' '.join(marks)}")
    print(f"  {'Legend:':>15s} ★ ≥.25  ● ≥.10  ○ ≥.05  · <.05")

    print(f"\nOutput mask:")
    header = "".join(f"{n:>8s}" for n in names)
    print(f"  {'':15s}{header}")
    vals = "".join(f"{v:8.2f}" for v in MODULE_PRESET['output_mask'])
    print(f"  {'→ muscles':15s}{vals}")

    pathways = []
    for i in range(n_mod):
        for j in range(n_mod):
            if i != j and conn[i][j] >= 0.05:
                pathways.append((names[i], names[j], conn[i][j]))
    pathways.sort(key=lambda x: -x[2])

    print(f"\nKey inter-module pathways (sorted by strength):")
    for src, tgt, prob in pathways:
        print(f"  {src}→{tgt:15s} p={prob:.2f}")

    print("=" * 70)
    print()


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a 4-module reaching model (PMd, M1, S1, SC)."""
    name: str
    n_modules: int = 4
    episode_duration: float = 3.0
    proprioception_delay: float = 0.02
    vision_delay: float = 0.08
    proprioception_noise: float = 1e-3
    vision_noise: float = 1e-3
    action_noise: float = 1e-4
    learning_rate: float = 1e-3

    # Module parameters (defaults from 4-module preset)
    module_names: list = field(default_factory=lambda: ["premotor", "motor", "somatosensory", "spinal"])
    module_sizes: list = field(default_factory=lambda: [256, 256, 128, 64])
    vision_mask: list = field(default_factory=lambda:  [0.50, 0.15, 0.00, 0.00])
    proprio_mask: list = field(default_factory=lambda: [0.00, 0.10, 0.40, 0.50])
    task_mask: list = field(default_factory=lambda:    [0.50, 0.10, 0.00, 0.00])
    connectivity_mask: list = field(default_factory=lambda: [
        [0.70, 0.35, 0.02, 0.08],
        [0.15, 0.70, 0.05, 0.30],
        [0.12, 0.25, 0.70, 0.05],
        [0.02, 0.08, 0.15, 0.70],
    ])
    output_mask: list = field(default_factory=lambda: [0.00, 0.00, 0.00, 0.50])
    spectral_scaling: float = 1.15

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = d.copy()
        # Strip legacy fields that are no longer part of the dataclass
        d.pop('modular', None)
        d.pop('n_units', None)
        # Infer n_modules from module_sizes if missing
        if 'n_modules' not in d:
            d['n_modules'] = len(d.get('module_sizes', [256, 256, 128, 64]))
        return cls(**d)


@dataclass
class TrainingState:
    """Tracks training progress."""
    batches_completed: int = 0
    loss_history: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "batches_completed": self.batches_completed,
            "loss_history": self.loss_history
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingState":
        return cls(
            batches_completed=d.get("batches_completed", 0),
            loss_history=d.get("loss_history", {})
        )


# =============================================================================
# Main Model Class
# =============================================================================

class ReachingModel:
    """
    A 4-module neural network model that learns to control a simulated arm
    for reaching tasks (PMd, M1, S1, SC).

    Attributes:
        name: The model's name (also used for the save directory)
        config: Model configuration parameters
        env: The motornet environment
        task: The reaching task generator
        policy: The modular GRU network
        training_state: Tracks training progress and loss history
    """

    def __init__(
        self,
        name: str,
        config: ModelConfig,
        env: ExperimentEnv,
        task: ExperimentTask,
        policy: th.nn.Module,
        training_state: Optional[TrainingState] = None,
        device: th.device = None
    ):
        self.name = name
        self.config = config
        self.env = env
        self.task = task
        self.policy = policy
        self.training_state = training_state or TrainingState()
        self.device = device or th.device("cpu")
        self._optimizer = None

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        module_sizes: Optional[list] = None,
        episode_duration: float = 3.0,
        save: bool = True,
        **kwargs
    ) -> "ReachingModel":
        """
        Create a new 4-module reaching model (PMd, M1, S1, SC).

        Args:
            name: Name for the model (will create a directory with this name)
            module_sizes: Override default module sizes [256, 256, 128, 64]
            episode_duration: Duration of each simulation episode in seconds
            save: If True, save the model after creation
            **kwargs: Override any config parameter (e.g. vision_delay=0.10)

        Returns:
            A new ReachingModel instance

        Example:
            model = ReachingModel.create("my_model")
            model = ReachingModel.create("big", module_sizes=[512, 256, 128, 64])
        """
        device = th.device("cpu")

        # Build configuration from preset + overrides
        preset = MODULE_PRESET.copy()

        if module_sizes is not None:
            if len(module_sizes) != 4:
                raise ValueError(f"Expected 4 module sizes, got {len(module_sizes)}")
            preset['module_sizes'] = module_sizes

        # User kwargs override preset values
        for key in list(kwargs.keys()):
            if key in preset:
                preset[key] = kwargs.pop(key)

        config = ModelConfig(
            name=name,
            episode_duration=episode_duration,
            **preset,
            **kwargs
        )

        # Create the biomechanical arm model
        effector = mn.effector.RigidTendonArm26(
            muscle=mn.muscle.RigidTendonHillMuscle()
        )

        # Create the environment
        env = ExperimentEnv(
            effector=effector,
            max_ep_duration=config.episode_duration,
            proprioception_delay=config.proprioception_delay,
            vision_delay=config.vision_delay,
            proprioception_noise=config.proprioception_noise,
            vision_noise=config.vision_noise,
            action_noise=config.action_noise
        )

        # Create the task
        task = ExperimentTask(effector=env.effector)

        # Get input dimensions by generating a sample
        n_t = int(config.episode_duration / env.effector.dt)
        inputs, _, _, _, _ = task.generate(1, n_t)
        n_task_inputs = inputs['inputs'].shape[2]
        total_input_size = env.observation_space.shape[0] + n_task_inputs

        # Calculate input dimension indices for the modular network
        task_dim = np.arange(inputs['inputs'].shape[-1])
        vision_dim = np.arange(env.get_vision().shape[1]) + task_dim[-1] + 1
        proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1

        policy = ModularPolicyGRU(
            input_size=total_input_size,
            module_size=config.module_sizes,
            output_size=env.n_muscles,
            vision_dim=vision_dim,
            proprio_dim=proprio_dim,
            task_dim=task_dim,
            vision_mask=config.vision_mask,
            proprio_mask=config.proprio_mask,
            task_mask=config.task_mask,
            connectivity_mask=np.array(config.connectivity_mask),
            output_mask=config.output_mask,
            connectivity_delay=np.zeros((len(config.module_sizes), len(config.module_sizes))),
            spectral_scaling=config.spectral_scaling,
            device=device,
            activation='tanh'
        )

        model = cls(
            name=name,
            config=config,
            env=env,
            task=task,
            policy=policy,
            device=device
        )

        if save:
            model.save()
            print(f"Created and saved model '{name}'")

        return model

    @classmethod
    def load(cls, name: str) -> "ReachingModel":
        """
        Load a previously saved model from disk.

        Args:
            name: Name of the model (directory name)

        Returns:
            The loaded ReachingModel instance

        Example:
            model = ReachingModel.load("my_model")
        """
        model_dir = name
        config_file = os.path.join(model_dir, f"{name}_config.json")
        weights_file = os.path.join(model_dir, f"{name}_weights.pkl")
        state_file = os.path.join(model_dir, f"{name}_training_state.json")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Model '{name}' not found. Looking for: {config_file}")

        # Load configuration
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)

        device = th.device("cpu")

        # Recreate environment
        effector = mn.effector.RigidTendonArm26(
            muscle=mn.muscle.RigidTendonHillMuscle()
        )
        env = ExperimentEnv(
            effector=effector,
            max_ep_duration=config.episode_duration,
            proprioception_delay=config.proprioception_delay,
            vision_delay=config.vision_delay,
            proprioception_noise=config.proprioception_noise,
            vision_noise=config.vision_noise,
            action_noise=config.action_noise
        )

        # Recreate task
        task = ExperimentTask(effector=env.effector)

        # Get input dimensions
        n_t = int(config.episode_duration / env.effector.dt)
        inputs, _, _, _, _ = task.generate(1, n_t)
        n_task_inputs = inputs['inputs'].shape[2]
        total_input_size = env.observation_space.shape[0] + n_task_inputs

        # Recreate policy
        task_dim = np.arange(inputs['inputs'].shape[-1])
        vision_dim = np.arange(env.get_vision().shape[1]) + task_dim[-1] + 1
        proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1

        policy = ModularPolicyGRU(
            input_size=total_input_size,
            module_size=config.module_sizes,
            output_size=env.n_muscles,
            vision_dim=vision_dim,
            proprio_dim=proprio_dim,
            task_dim=task_dim,
            vision_mask=config.vision_mask,
            proprio_mask=config.proprio_mask,
            task_mask=config.task_mask,
            connectivity_mask=np.array(config.connectivity_mask),
            output_mask=config.output_mask,
            connectivity_delay=np.zeros((len(config.module_sizes), len(config.module_sizes))),
            spectral_scaling=config.spectral_scaling,
            device=device,
            activation='tanh'
        )

        # Load weights if they exist
        if os.path.exists(weights_file):
            weights = th.load(weights_file, weights_only=True)
            policy.load_state_dict(weights)

        # Load training state if it exists
        training_state = TrainingState()
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                training_state = TrainingState.from_dict(json.load(f))

        model = cls(
            name=name,
            config=config,
            env=env,
            task=task,
            policy=policy,
            training_state=training_state,
            device=device
        )

        print(f"Loaded model '{name}' (batches trained: {training_state.batches_completed})")
        return model

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    def train(
        self,
        n_batches: int = 10000,
        batch_size: int = 64,
        ff_strength: float = 0.0,
        task_mode: Literal["random", "center_out"] = "random",
        plot_interval: int = 100,
        quiet: bool = False
    ) -> dict:
        """
        Train the model on reaching movements.

        Args:
            n_batches: Number of training batches
            batch_size: Number of trials per batch
            ff_strength: Force field strength (0 = no force field)
            task_mode: "random" for random reaches, "center_out" for center-out reaches
            plot_interval: How often to save intermediate plots (0 = no plots)
            quiet: If True, suppress progress bar

        Returns:
            Dictionary containing the loss history

        Example:
            model.train(n_batches=10000, batch_size=64)
            model.train(n_batches=5000, ff_strength=15.0)  # Train with force field
        """
        # Ensure output directory exists
        os.makedirs(self.name, exist_ok=True)

        # Set task mode
        if task_mode == "random":
            self.task.run_mode = 'train'
        else:
            self.task.run_mode = 'train_center_out'

        # Setup optimizer
        if self._optimizer is None:
            self._optimizer = th.optim.Adam(
                self.policy.parameters(),
                lr=self.config.learning_rate
            )

        # Calculate timesteps
        n_t = int(self.config.episode_duration / self.env.effector.dt)

        # Loss function and keys
        calculate_loss = lambda ep: calculate_loss_mehrdad(ep, self.policy, self.env)
        loss_keys = ["total", "pos", "act", "force", "force_diff", "hdn", "hdn_diff", "weight_decay", "speed", "hdn_jerk"]

        # Initialize loss history if needed
        if not self.training_state.loss_history:
            self.training_state.loss_history = {key: [] for key in loss_keys}

        # Training loop
        iterator = range(n_batches)
        if not quiet:
            iterator = tqdm(iterator, desc=f"Training '{self.name}'", unit="batch")

        for i in iterator:
            # Run episode
            episode_data = run_episode(
                self.env, self.task, self.policy,
                batch_size, n_t, self.device, k=ff_strength
            )

            # Calculate loss and backpropagate
            loss = calculate_loss(episode_data)
            loss['total'].backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

            # Track losses
            with th.no_grad():
                for key in loss_keys:
                    self.training_state.loss_history[key].append(loss[key].item())

            self.training_state.batches_completed += 1

            # Intermediate plots
            if plot_interval > 0 and (i + 1) % plot_interval == 0:
                self._save_training_plots(episode_data, i + 1, batch_size)
                self.save()

        # Final plots and save
        self._save_training_plots(episode_data, n_batches, batch_size)
        self.save()

        if not quiet:
            print(f"Training complete. Total batches: {self.training_state.batches_completed}")

        return self.training_state.loss_history

    def test(
        self,
        n_targets: int = 8,
        ff_strength: float = 0.0,
        simulation_time: float = None,
        save_plots: bool = True,
        save_data: bool = True
    ) -> dict:
        """
        Test the model on center-out reaching movements.

        Args:
            n_targets: Number of targets arranged in a circle
            ff_strength: Force field strength (0 = no force field)
            simulation_time: Duration of simulation (default: episode_duration)
            save_plots: If True, save plots to the model directory
            save_data: If True, save episode data to disk

        Returns:
            Dictionary containing episode data

        Example:
            results = model.test(n_targets=8)
            results = model.test(n_targets=8, ff_strength=15.0)  # Test with force field
        """
        os.makedirs(self.name, exist_ok=True)

        if simulation_time is None:
            simulation_time = self.config.episode_duration

        n_t = int(simulation_time / self.env.dt)
        self.task.run_mode = 'test_center_out'

        print(f"Testing '{self.name}' on center-out task ({n_targets} targets, FF={ff_strength})")

        # Run test episode
        episode_data = run_episode(
            self.env, self.task, self.policy,
            n_targets, n_t, self.device, k=ff_strength
        )

        ff_suffix = f"_ff{ff_strength:.0f}" if ff_strength > 0 else ""

        if save_plots:
            # Plot hand paths
            plot_handpaths(
                episode_data=episode_data,
                fname=os.path.join(self.name, f"{self.name}_test_handpaths{ff_suffix}.png"),
                figtitle=f"Test: {n_targets} targets (FF={ff_strength})"
            )

            # Plot individual trials
            for i in range(n_targets):
                plot_signals(
                    episode_data=episode_data,
                    fname=os.path.join(self.name, f"{self.name}_test_trial_{i}{ff_suffix}.png"),
                    figtitle=f"Test trial {i}",
                    trial=i
                )

            print(f"Saved test plots to '{self.name}/'")

        if save_data:
            data_file = os.path.join(self.name, f"{self.name}_test_data{ff_suffix}.pkl")
            # Convert tensors to numpy for saving
            save_data_dict = {
                k: v.detach().cpu().numpy() if th.is_tensor(v) else v
                for k, v in episode_data.items()
            }
            with open(data_file, 'wb') as f:
                pickle.dump(save_data_dict, f)
            print(f"Saved test data to '{data_file}'")

        return episode_data

    def save(self, name: str = None) -> None:
        """
        Save the model to disk.

        Args:
            name: Optional new name for the model. If not provided, uses current name.

        Example:
            model.save()
            model.save("my_model_backup")
        """
        if name is not None:
            self.name = name
            self.config.name = name

        model_dir = self.name
        os.makedirs(model_dir, exist_ok=True)

        # Save configuration
        config_file = os.path.join(model_dir, f"{self.name}_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        weights_file = os.path.join(model_dir, f"{self.name}_weights.pkl")
        th.save(self.policy.state_dict(), weights_file)

        # Save training state
        state_file = os.path.join(model_dir, f"{self.name}_training_state.json")
        with open(state_file, 'w') as f:
            json.dump(self.training_state.to_dict(), f, indent=2)

        print(f"Saved model to '{model_dir}/'")

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _save_training_plots(self, episode_data, batch_num, batch_size):
        """Save intermediate training plots."""
        plot_losses(
            loss_history=self.training_state.loss_history,
            fname=os.path.join(self.name, f"{self.name}_losses.png")
        )
        plot_handpaths(
            episode_data=episode_data,
            fname=os.path.join(self.name, f"{self.name}_handpaths.png"),
            figtitle=f"Batch {batch_num} (n={batch_size})"
        )
        for j in range(min(4, episode_data['xy'].shape[0])):
            plot_signals(
                episode_data=episode_data,
                fname=os.path.join(self.name, f"{self.name}_signals_{j}.png"),
                figtitle=f"Batch {batch_num} (n={batch_size})",
                trial=j
            )

    def summary(self) -> str:
        """Return a summary of the model."""
        modules_detail = ", ".join(
            f"{n}({s})" for n, s in zip(self.config.module_names, self.config.module_sizes)
        )
        return (
            f"ReachingModel '{self.name}'\n"
            f"  Architecture: [{modules_detail}]\n"
            f"  Episode duration: {self.config.episode_duration}s\n"
            f"  Batches trained: {self.training_state.batches_completed}\n"
            f"  Directory: {self.name}/"
        )

    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line interface for ReachingModel."""
    parser = argparse.ArgumentParser(
        description="Train and test 4-module neural networks for arm reaching movements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create my_model
  %(prog)s train my_model --batches 10000
  %(prog)s test my_model --targets 8 --ff 15.0
  %(prog)s info my_model
  %(prog)s arch
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # CREATE command
    create_parser = subparsers.add_parser("create", help="Create a new model")
    create_parser.add_argument("name", help="Name for the model")

    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train an existing model")
    train_parser.add_argument("name", help="Name of the model to train")
    train_parser.add_argument("--batches", type=int, default=10000, help="Number of training batches (default: 10000)")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    train_parser.add_argument("--ff", type=float, default=0.0, help="Force field strength (default: 0.0)")
    train_parser.add_argument("--task", choices=["random", "center_out"], default="random",
                              help="Training task type (default: random)")

    # TEST command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("name", help="Name of the model to test")
    test_parser.add_argument("--targets", type=int, default=8, help="Number of targets (default: 8)")
    test_parser.add_argument("--ff", type=float, default=0.0, help="Force field strength (default: 0.0)")

    # INFO command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("name", help="Name of the model")

    # SAVE command (rename/copy)
    save_parser = subparsers.add_parser("save", help="Save model with a new name")
    save_parser.add_argument("name", help="Current model name")
    save_parser.add_argument("new_name", help="New name for the model")

    # ARCH command — show architecture details
    arch_parser = subparsers.add_parser("arch", help="Show architecture details")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "create":
        ReachingModel.create(name=args.name)

    elif args.command == "train":
        model = ReachingModel.load(args.name)
        model.train(
            n_batches=args.batches,
            batch_size=args.batch_size,
            ff_strength=args.ff,
            task_mode=args.task,
        )

    elif args.command == "test":
        model = ReachingModel.load(args.name)
        model.test(
            n_targets=args.targets,
            ff_strength=args.ff,
        )

    elif args.command == "info":
        model = ReachingModel.load(args.name)
        print(model.summary())

    elif args.command == "save":
        model = ReachingModel.load(args.name)
        model.save(args.new_name)

    elif args.command == "arch":
        print_architecture()


if __name__ == "__main__":
    main()
