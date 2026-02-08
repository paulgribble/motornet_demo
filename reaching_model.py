"""
Reaching Model - A simplified interface for training RNNs to control biomechanical arm models.

This module provides a high-level API for creating, training, testing, and saving neural network
models that learn to control a simulated arm to perform reaching movements.

Example usage (API):
    from reaching_model import ReachingModel

    # Create and train a new model
    model = ReachingModel.create("my_model", n_units=256)
    model.train(n_batches=10000, batch_size=64)
    model.test(n_targets=8)
    model.save()

    # Load an existing model
    model = ReachingModel.load("my_model")
    model.test(n_targets=8, ff_strength=15.0)

Example usage (CLI):
    uv run reaching_model.py create my_model --units 256
    uv run reaching_model.py train my_model --batches 10000 --batch-size 64
    uv run reaching_model.py test my_model --targets 8 --ff 15.0
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
os.environ['OMP_NUM_THREADS'] = '1'
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
from my_policy import Policy, ModularPolicyGRU
from my_loss import calculate_loss_michaels, calculate_loss_mehrdad
from my_utils import run_episode, plot_losses, plot_handpaths, plot_signals, plot_signals_modular


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a reaching model."""
    name: str
    n_units: int = 256
    modular: bool = False
    module_sizes: list = field(default_factory=lambda: [256, 256, 128, 64])
    module_names: list = field(default_factory=lambda: ["premotor", "motor", "somatosensory", "spinal"])
    episode_duration: float = 3.0
    proprioception_delay: float = 0.02
    vision_delay: float = 0.07
    proprioception_noise: float = 1e-3
    vision_noise: float = 1e-3
    action_noise: float = 1e-4
    learning_rate: float = 1e-3

    # Modular-specific parameters (used only when modular=True)
    vision_mask: list = field(default_factory=lambda: [0.2, 0.0, 0.0, 0.0])
    proprio_mask: list = field(default_factory=lambda: [0.0, 0.0, 0.5, 0.3])
    task_mask: list = field(default_factory=lambda: [0.2, 0.02, 0.0, 0.0])
    connectivity_mask: list = field(default_factory=lambda: [
        [1.0, 0.1, 0.05, 0.0],
        [0.2, 1.0, 0.2, 0.0],
        [0.05, 0.1, 1.0, 0.1],
        [0.0, 0.2, 0.05, 1.0]
    ])
    output_mask: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.5])
    spectral_scaling: float = 1.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
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
    A neural network model that learns to control a simulated arm for reaching tasks.

    This class wraps all the complexity of motornet, providing a simple interface
    for creating, training, testing, and saving models.

    Attributes:
        name: The model's name (also used for the save directory)
        config: Model configuration parameters
        env: The motornet environment
        task: The reaching task generator
        policy: The neural network (GRU or ModularGRU)
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
        n_units: int = 256,
        modular: bool = False,
        module_sizes: Optional[list] = None,
        episode_duration: float = 3.0,
        save: bool = True,
        **kwargs
    ) -> "ReachingModel":
        """
        Create a new reaching model with the specified architecture.

        Args:
            name: Name for the model (will create a directory with this name)
            n_units: Number of hidden units (for simple models)
            modular: If True, create a modular architecture with multiple RNN modules
            module_sizes: List of sizes for each module (modular only). Default: [256, 256, 128, 64]
            episode_duration: Duration of each simulation episode in seconds
            save: If True, save the model after creation
            **kwargs: Additional configuration parameters

        Returns:
            A new ReachingModel instance

        Example:
            # Simple model with 256 units
            model = ReachingModel.create("my_model", n_units=256)

            # Modular model with 4 modules (premotor, motor, somatosensory, spinal)
            model = ReachingModel.create("modular_model", modular=True)
        """
        device = th.device("cpu")

        # Build configuration
        if module_sizes is None:
            module_sizes = [256, 256, 128, 64]

        config = ModelConfig(
            name=name,
            n_units=n_units,
            modular=modular,
            module_sizes=module_sizes,
            episode_duration=episode_duration,
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
        inputs, _, _, _ = task.generate(1, n_t)
        n_task_inputs = inputs['inputs'].shape[2]
        total_input_size = env.observation_space.shape[0] + n_task_inputs

        # Create the policy network
        if modular:
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
        else:
            policy = Policy(
                input_dim=total_input_size,
                hidden_dim=n_units,
                output_dim=env.n_muscles,
                device=device
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
        inputs, _, _, _ = task.generate(1, n_t)
        n_task_inputs = inputs['inputs'].shape[2]
        total_input_size = env.observation_space.shape[0] + n_task_inputs

        # Recreate policy
        if config.modular:
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
        else:
            policy = Policy(
                input_dim=total_input_size,
                hidden_dim=config.n_units,
                output_dim=env.n_muscles,
                device=device
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

        # Choose loss function based on model type
        if self.config.modular:
            calculate_loss = lambda ep: calculate_loss_mehrdad(ep, self.policy, self.env)
            plot_signals_fn = plot_signals_modular
            loss_keys = ["total", "pos", "act", "force", "force_diff", "hdn", "hdn_diff", "weight_decay", "speed", "hdn_jerk"]
        else:
            calculate_loss = calculate_loss_michaels
            plot_signals_fn = plot_signals
            loss_keys = ["total", "position", "speed", "jerk", "muscle", "muscle_derivative", "hidden", "hidden_derivative"]

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
                self._save_training_plots(episode_data, i + 1, batch_size, plot_signals_fn)

        # Final plots and save
        self._save_training_plots(episode_data, n_batches, batch_size, plot_signals_fn)
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

        # Choose appropriate plotting function
        plot_signals_fn = plot_signals_modular if self.config.modular else plot_signals

        if save_plots:
            # Plot hand paths
            ff_suffix = f"_ff{ff_strength:.0f}" if ff_strength > 0 else ""
            plot_handpaths(
                episode_data=episode_data,
                fname=os.path.join(self.name, f"{self.name}_test_handpaths{ff_suffix}.png"),
                figtitle=f"Test: {n_targets} targets (FF={ff_strength})"
            )

            # Plot individual trials
            for i in range(n_targets):
                plot_signals_fn(
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

    def _save_training_plots(self, episode_data, batch_num, batch_size, plot_signals_fn):
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
            plot_signals_fn(
                episode_data=episode_data,
                fname=os.path.join(self.name, f"{self.name}_signals_{j}.png"),
                figtitle=f"Batch {batch_num} (n={batch_size})",
                trial=j
            )

    def summary(self) -> str:
        """Return a summary of the model."""
        arch_type = "Modular" if self.config.modular else "Simple"
        if self.config.modular:
            names = getattr(self.config, 'module_names', None)
            if not names or len(names) != len(self.config.module_sizes):
                names = [f"module_{i}" for i in range(len(self.config.module_sizes))]
            modules_detail = ", ".join(f"{n}({s})" for n, s in zip(names, self.config.module_sizes))
            units_str = f"modules: [{modules_detail}]"
        else:
            units_str = f"{self.config.n_units} units"

        return (
            f"ReachingModel '{self.name}'\n"
            f"  Architecture: {arch_type} GRU ({units_str})\n"
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
        description="Train and test neural networks for arm reaching movements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create a new model:
    %(prog)s create my_model --units 256
    %(prog)s create modular_model --modular

  Train a model:
    %(prog)s train my_model --batches 10000 --batch-size 64
    %(prog)s train my_model --batches 5000 --ff 15.0

  Test a model:
    %(prog)s test my_model --targets 8
    %(prog)s test my_model --targets 8 --ff 15.0

  Show model info:
    %(prog)s info my_model
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # CREATE command
    create_parser = subparsers.add_parser("create", help="Create a new model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Simple model with custom delays:
    %(prog)s my_model --proprio-delay 0.04 --vision-delay 0.12

  Model with high noise:
    %(prog)s noisy --proprio-noise 0.005 --vision-noise 0.003

  Modular model with custom connectivity:
    %(prog)s modular --modular --module-sizes 200 100 50 \\
        --vision-mask 0.3 0.1 0.0 --proprio-mask 0.0 0.1 0.6
        """)
    create_parser.add_argument("name", help="Name for the model")

    # Architecture options
    arch_group = create_parser.add_argument_group("Architecture")
    arch_group.add_argument("--units", type=int, default=256,
                            help="Number of hidden units for simple model (default: 256)")
    arch_group.add_argument("--modular", action="store_true",
                            help="Create a modular architecture with multiple RNN modules")
    arch_group.add_argument("--module-sizes", type=int, nargs="+", default=[256, 256, 128, 64],
                            help="Sizes of each module for modular architecture (default: 256 256 128 64)")

    # Episode options
    ep_group = create_parser.add_argument_group("Episode settings")
    ep_group.add_argument("--duration", type=float, default=3.0,
                          help="Episode duration in seconds (default: 3.0)")

    # Sensory delay options
    delay_group = create_parser.add_argument_group("Sensory delays (seconds)")
    delay_group.add_argument("--proprio-delay", type=float, default=0.02,
                             help="Proprioceptive feedback delay (default: 0.02)")
    delay_group.add_argument("--vision-delay", type=float, default=0.07,
                             help="Visual feedback delay (default: 0.07)")

    # Noise options
    noise_group = create_parser.add_argument_group("Noise levels (std dev)")
    noise_group.add_argument("--proprio-noise", type=float, default=1e-3,
                             help="Proprioceptive noise (default: 0.001)")
    noise_group.add_argument("--vision-noise", type=float, default=1e-3,
                             help="Visual noise (default: 0.001)")
    noise_group.add_argument("--action-noise", type=float, default=1e-4,
                             help="Motor command noise (default: 0.0001)")

    # Learning options
    learn_group = create_parser.add_argument_group("Learning")
    learn_group.add_argument("--learning-rate", type=float, default=1e-3,
                             help="Optimizer learning rate (default: 0.001)")

    # Modular-specific options
    mod_group = create_parser.add_argument_group("Modular architecture options (only used with --modular)")
    mod_group.add_argument("--vision-mask", type=float, nargs="+", default=[0.2, 0.0, 0.0, 0.0],
                           help="Vision input probability per module (default: 0.2 0.0 0.0 0.0)")
    mod_group.add_argument("--proprio-mask", type=float, nargs="+", default=[0.0, 0.0, 0.5, 0.3],
                           help="Proprioception input probability per module (default: 0.0 0.0 0.5 0.3)")
    mod_group.add_argument("--task-mask", type=float, nargs="+", default=[0.2, 0.02, 0.0, 0.0],
                           help="Task input probability per module (default: 0.2 0.02 0.0 0.0)")
    mod_group.add_argument("--output-mask", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.5],
                           help="Output probability per module (default: 0.0 0.0 0.0 0.5)")
    mod_group.add_argument("--spectral-scaling", type=float, default=1.1,
                           help="Spectral radius scaling for recurrent weights (default: 1.1)")

    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train an existing model")
    train_parser.add_argument("name", help="Name of the model to train")
    train_parser.add_argument("--batches", type=int, default=10000, help="Number of training batches (default: 10000)")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    train_parser.add_argument("--ff", type=float, default=0.0, help="Force field strength (default: 0.0)")
    train_parser.add_argument("--task", choices=["random", "center_out"], default="random",
                              help="Training task type (default: random)")
    train_parser.add_argument("--plot-interval", type=int, default=100,
                              help="Batches between plot updates (default: 100, 0 to disable)")
    train_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # TEST command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("name", help="Name of the model to test")
    test_parser.add_argument("--targets", type=int, default=8, help="Number of targets (default: 8)")
    test_parser.add_argument("--ff", type=float, default=0.0, help="Force field strength (default: 0.0)")
    test_parser.add_argument("--no-plots", action="store_true", help="Don't save plots")
    test_parser.add_argument("--no-data", action="store_true", help="Don't save episode data")

    # INFO command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("name", help="Name of the model")

    # SAVE command (rename/copy)
    save_parser = subparsers.add_parser("save", help="Save model with a new name")
    save_parser.add_argument("name", help="Current model name")
    save_parser.add_argument("new_name", help="New name for the model")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "create":
        ReachingModel.create(
            name=args.name,
            n_units=args.units,
            modular=args.modular,
            module_sizes=args.module_sizes,
            episode_duration=args.duration,
            proprioception_delay=args.proprio_delay,
            vision_delay=args.vision_delay,
            proprioception_noise=args.proprio_noise,
            vision_noise=args.vision_noise,
            action_noise=args.action_noise,
            learning_rate=args.learning_rate,
            vision_mask=args.vision_mask,
            proprio_mask=args.proprio_mask,
            task_mask=args.task_mask,
            output_mask=args.output_mask,
            spectral_scaling=args.spectral_scaling
        )

    elif args.command == "train":
        model = ReachingModel.load(args.name)
        model.train(
            n_batches=args.batches,
            batch_size=args.batch_size,
            ff_strength=args.ff,
            task_mode=args.task,
            plot_interval=args.plot_interval,
            quiet=args.quiet
        )

    elif args.command == "test":
        model = ReachingModel.load(args.name)
        model.test(
            n_targets=args.targets,
            ff_strength=args.ff,
            save_plots=not args.no_plots,
            save_data=not args.no_data
        )

    elif args.command == "info":
        model = ReachingModel.load(args.name)
        print(model.summary())

    elif args.command == "save":
        model = ReachingModel.load(args.name)
        model.save(args.new_name)


if __name__ == "__main__":
    main()
