"""
Reaching Model - A simplified interface for training modular RNNs to control a biomechanical arm.

The default architecture is a 2-module modular GRU (M1 + SC):
  M1 (motor cortex) → SC (spinal cord)

M1 receives task inputs (target, go cue) and vision, and generates descending
commands to SC. SC receives proprioceptive signals and descending commands from
M1, and drives muscle output with a 1-timestep delay.

Larger architectures (3-module, 4-module) or smaller (1-module) can be configured via parameters.
See go_1module.py, go_3module.py, and go_4module.py for examples.

Example usage:
    from reaching_model import ReachingModel

    model = ReachingModel.create("my_model")
    model.train(n_batches=10000, batch_size=64)
    model.test(n_targets=8)
    model.save()
"""

from __future__ import annotations

import os
import sys
import json
import pickle
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
from my_loss import michaels_modular_loss
from my_utils import run_episode, plot_losses, plot_handpaths, plot_signals


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a modular reaching model."""
    name: str
    n_modules: int = 2
    episode_duration: float = 3.0
    proprioception_delay: float = 0.01
    vision_delay: float = 0.11
    proprioception_noise: float = 1e-3
    vision_noise: float = 1e-3
    action_noise: float = 1e-4
    learning_rate: float = 1e-3
    activation: str = 'rect_tanh' # 'tanh' or 'rect_tanh'
    output_delay: int = 1

    # Module parameters (defaults for 2-module: M1 + SC)
    module_names: list = field(default_factory=lambda: ["motor", "spinal"])
    module_sizes: list = field(default_factory=lambda: [128, 32])
    vision_mask: list = field(default_factory=lambda:  [1.0, 0.0])
    proprio_mask: list = field(default_factory=lambda: [0.0, 1.0])
    task_mask: list = field(default_factory=lambda:    [1.0, 0.0])
    connectivity_mask: list = field(default_factory=lambda: [
        [0.7, 0.1], # motor receives from itself (0.7) and spinal (0.1)
        [0.5, 0.7], # spinal receives from motor (0.5) and itself (0.7)
    ])
    output_mask: list = field(default_factory=lambda: [0.0, 1.0])
    spectral_scaling: float = 1.30

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = d.copy()
        # Infer n_modules from module_sizes if missing
        if 'n_modules' not in d:
            d['n_modules'] = len(d.get('module_sizes', [128, 32]))
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
    A modular neural network model that learns to control a simulated arm
    for reaching tasks. Default architecture: M1 + SC (2-module).

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

    @staticmethod
    def _build_components(config):
        """Create env, task, and policy from a ModelConfig."""
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
        task = ExperimentTask(effector=env.effector)

        n_t = int(config.episode_duration / env.effector.dt)
        inputs, _, _, _, _ = task.generate(1, n_t)
        n_task_inputs = inputs['inputs'].shape[2]
        total_input_size = env.observation_space.shape[0] + n_task_inputs

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
            device=th.device("cpu"),
            activation=config.activation,
            output_delay=config.output_delay,
        )
        return env, task, policy

    @classmethod
    def create(
        cls,
        name: str,
        episode_duration: float = 3.0,
        save: bool = True,
        **kwargs
    ) -> "ReachingModel":
        """
        Create a new reaching model.

        Args:
            name: Name for the model (will create a directory with this name)
            episode_duration: Duration of each simulation episode in seconds
            save: If True, save the model after creation
            **kwargs: Override any ModelConfig parameter (e.g. module_sizes, vision_mask)

        Returns:
            A new ReachingModel instance

        Example:
            model = ReachingModel.create("my_model")
            model = ReachingModel.create("big", module_sizes=[256, 256, 128, 32])
        """
        config = ModelConfig(
            name=name,
            episode_duration=episode_duration,
            **kwargs
        )

        env, task, policy = cls._build_components(config)

        model = cls(
            name=name,
            config=config,
            env=env,
            task=task,
            policy=policy,
            device=th.device("cpu")
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

        env, task, policy = cls._build_components(config)

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
            device=th.device("cpu")
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
        quiet: bool = False,
        tqdm_position: int = None,
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
        calculate_loss = michaels_modular_loss
        loss_keys =calculate_loss(episode_data=None, returnKeys=True)

        # Initialize loss history if needed
        if not self.training_state.loss_history:
            self.training_state.loss_history = {key: [] for key in loss_keys}

        # Training loop
        iterator = range(n_batches)
        if not quiet:
            iterator = tqdm(iterator, desc=f"Training '{self.name}'", unit="batch",
                            position=tqdm_position, leave=True)

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
                self.save(quiet=True)

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

    def save(self, name: str = None, quiet: bool = False) -> None:
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

        if not quiet:
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
