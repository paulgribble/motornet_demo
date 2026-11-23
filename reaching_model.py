"""
ReachingModel: build, train, test, save and load a modular RNN that controls a simulated arm.

The default architecture has two modules, M1 (motor cortex) -> SC (spinal cord):
M1 receives the task inputs (target, go cue) and vision; SC receives proprioception
and the descending commands from M1, and drives the muscles.
See go_1module.py ... go_4module.py for other architectures.

Example usage:
    from reaching_model import ReachingModel

    model = ReachingModel.create("my_model")
    model.train(n_batches=10000, batch_size=64)
    model.test(n_targets=8)

    model = ReachingModel.load("my_model")   # later: load it again
"""

import os
import sys
import json
import pickle
from dataclasses import dataclass, field, asdict

# Use a single thread (set BEFORE importing torch/numpy)
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
from tqdm import tqdm

import motornet as mn
from my_env import ExperimentEnv
from my_task import ExperimentTask
from my_policy import ModularPolicyGRU
from my_loss import michaels_modular_loss as calculate_loss
from my_utils import run_episode, model_path
from my_plots import plot_losses, plot_handpaths, plot_signals


@dataclass
class ModelConfig:
    """Everything needed to rebuild a model (saved as <name>_config.json)."""
    name: str
    n_modules: int = 2            # always set from module_sizes (see __post_init__)
    episode_duration: float = 3.0 # seconds
    proprioception_delay: float = 0.01
    vision_delay: float = 0.11
    proprioception_noise: float = 1e-3
    vision_noise: float = 1e-3
    action_noise: float = 1e-4
    learning_rate: float = 1e-3
    # 'tanh' (-1 to 1) or 'rect_tanh' = max(0, tanh), which keeps hidden activity
    # in 0 to 1 so units behave like firing rates (never negative)
    activation: str = 'rect_tanh'
    output_delay: int = 1         # timesteps

    # One entry per module. The masks give the probability of a connection:
    # from each kind of input to each module, between modules, and from each module to the muscles.
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

    def __post_init__(self):
        self.n_modules = len(self.module_sizes)


@dataclass
class TrainingState:
    """Training progress (saved as <name>_training_state.json)."""
    batches_completed: int = 0
    loss_history: dict = field(default_factory=dict)


class ReachingModel:
    """A modular RNN (policy) controlling an arm (env) on a reaching task (task).

    All of a model's files live in a directory named after the model.
    """

    def __init__(self, config, training_state=None):
        self.config = config
        self.name = config.name
        self.training_state = training_state or TrainingState()
        self.device = th.device("cpu")
        self.env, self.task, self.policy = self._build_components(config)
        self.optimizer = th.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    @staticmethod
    def _build_components(config):
        """Create env, task, and policy from a ModelConfig."""
        effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
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

        # Which columns of the network input hold each kind of input.
        # run_episode feeds the network [vision, proprioception, task inputs], in that order.
        vision_dim = np.arange(env.get_vision().shape[1])
        proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1
        task_dim = np.arange(task.n_inputs) + proprio_dim[-1] + 1

        n_modules = len(config.module_sizes)
        policy = ModularPolicyGRU(
            input_size=env.observation_space.shape[0] + task.n_inputs,
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
            connectivity_delay=np.zeros((n_modules, n_modules)),
            spectral_scaling=config.spectral_scaling,
            device=th.device("cpu"),
            activation=config.activation,
            output_delay=config.output_delay,
        )
        return env, task, policy

    @classmethod
    def create(cls, name, **kwargs):
        """Create (and save) a new, untrained model. kwargs override any ModelConfig field,
        e.g. ReachingModel.create("big", module_sizes=[256, 64])."""
        model = cls(ModelConfig(name=name, **kwargs))
        model.save()
        print(f"Created and saved model '{name}'")
        return model

    @classmethod
    def load(cls, name):
        """Load a previously saved model from its directory."""
        with open(model_path(name, "config.json")) as f:
            config = ModelConfig(**json.load(f))
        with open(model_path(name, "training_state.json")) as f:
            training_state = TrainingState(**json.load(f))
        model = cls(config, training_state)
        model.policy.load_state_dict(th.load(model_path(name, "weights.pkl"), weights_only=True))
        print(f"Loaded model '{name}' (batches trained: {training_state.batches_completed})")
        return model

    def save(self, quiet=False):
        """Save configuration, weights and training state to the model's directory."""
        os.makedirs(self.name, exist_ok=True)
        with open(model_path(self.name, "config.json"), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        th.save(self.policy.state_dict(), model_path(self.name, "weights.pkl"))
        with open(model_path(self.name, "training_state.json"), 'w') as f:
            json.dump(asdict(self.training_state), f, indent=2)
        if not quiet:
            print(f"Saved model to '{self.name}/'")

    def train(self, n_batches=10000, batch_size=64, ff_strength=0.0, task_mode="random",
              plot_interval=100, quiet=False):
        """Train the model, saving plots and the model every plot_interval batches (0 = never).

        task_mode: "random" for random reaches, "center_out" for center-out reaches
        ff_strength: force field strength (0 = no force field)
        Calling train() again continues training. Returns the loss history.
        """
        self.task.run_mode = 'train' if task_mode == "random" else 'train_center_out'
        n_t = int(self.config.episode_duration / self.env.dt)
        loss_history = self.training_state.loss_history

        for i in tqdm(range(n_batches), desc=f"Training '{self.name}'", unit="batch", disable=quiet):
            episode_data = run_episode(self.env, self.task, self.policy, batch_size, n_t, self.device, k=ff_strength)
            loss = calculate_loss(episode_data)
            loss['total'].backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            for key, value in loss.items():
                loss_history.setdefault(key, []).append(value.item())
            self.training_state.batches_completed += 1

            if plot_interval > 0 and (i + 1) % plot_interval == 0:
                self._save_training_plots(episode_data, i + 1)
                self.save(quiet=True)

        self._save_training_plots(episode_data, n_batches)
        self.save()
        if not quiet:
            print(f"Training complete. Total batches: {self.training_state.batches_completed}")
        return loss_history

    def test(self, n_targets=8, ff_strength=0.0, save_plots=True, save_data=True):
        """Simulate center-out reaches to n_targets targets. Returns the episode data."""
        n_t = int(self.config.episode_duration / self.env.dt)
        self.task.run_mode = 'test_center_out'
        print(f"Testing '{self.name}' on center-out task ({n_targets} targets, FF={ff_strength})")

        episode_data = run_episode(self.env, self.task, self.policy, n_targets, n_t, self.device, k=ff_strength)

        ff_suffix = f"_ff{ff_strength:.0f}" if ff_strength > 0 else ""
        if save_plots:
            plot_handpaths(episode_data, fname=model_path(self.name, f"test_handpaths{ff_suffix}.png"),
                           figtitle=f"Test: {n_targets} targets (FF={ff_strength})")
            for i in range(n_targets):
                plot_signals(episode_data, fname=model_path(self.name, f"test_trial_{i}{ff_suffix}.png"),
                             figtitle=f"Test trial {i}", trial=i)
            print(f"Saved test plots to '{self.name}/'")

        if save_data:
            data_file = model_path(self.name, f"test_data{ff_suffix}.pkl")
            with open(data_file, 'wb') as f:
                pickle.dump({k: v.detach().cpu().numpy() if th.is_tensor(v) else v
                             for k, v in episode_data.items()}, f)
            print(f"Saved test data to '{data_file}'")

        return episode_data

    def _save_training_plots(self, episode_data, batch_num):
        """Plot the loss history and the most recent batch of trials."""
        batch_size = episode_data['xy'].shape[0]
        figtitle = f"Batch {batch_num} (n={batch_size})"
        plot_losses(self.training_state.loss_history, fname=model_path(self.name, "losses.png"))
        plot_handpaths(episode_data, fname=model_path(self.name, "handpaths.png"), figtitle=figtitle)
        for j in range(min(4, batch_size)):
            plot_signals(episode_data, fname=model_path(self.name, f"signals_{j}.png"), figtitle=figtitle, trial=j)

    def __repr__(self):
        modules = ", ".join(f"{n}({s})" for n, s in zip(self.config.module_names, self.config.module_sizes))
        return (
            f"ReachingModel '{self.name}'\n"
            f"  Architecture: [{modules}]\n"
            f"  Episode duration: {self.config.episode_duration}s\n"
            f"  Batches trained: {self.training_state.batches_completed}\n"
            f"  Directory: {self.name}/"
        )
