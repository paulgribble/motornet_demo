"""
Task generator for reaching experiments.

This module defines ExperimentTask, which generates trials for training
and testing neural networks on reaching movements.
"""

import numpy as np
import torch as th
from typing import Optional
from dataclasses import dataclass


__all__ = ["ExperimentTask", "TaskOutput"]


# =============================================================================
# Constants (defaults that can be overridden via constructor)
# =============================================================================

# Base joint configuration for center-out tasks [shoulder, elbow, shoulder_vel, elbow_vel]
DEFAULT_BASE_JOINT = np.deg2rad([50.0, 90.0, 0.0, 0.0]).astype(np.float32)

# Joint angle limits for random reach training (radians)
DEFAULT_JOINT_MIN = np.array([20, 20]) * np.pi / 180  # shoulder, elbow
DEFAULT_JOINT_MAX = np.array([110, 110]) * np.pi / 180

# Center-out task parameters
DEFAULT_CENTER_OUT_RADIUS = 0.10  # meters

# Timing defaults (seconds)
DEFAULT_DELAY_TARGET = (0.3, 0.8)  # when target appears
DEFAULT_DELAY_GO = (0.8, 1.3)      # when go cue appears

# Training parameters
DEFAULT_CATCH_TRIAL_PROBABILITY = 0.25
DEFAULT_INPUT_NOISE_STD = 1e-3

# Safety limit for random target search
MAX_TARGET_SEARCH_ITERATIONS = 1000


# =============================================================================
# Output data structure
# =============================================================================

@dataclass
class TaskOutput:
    """Output from task.generate().

    Attributes:
        inputs: Dict with 'inputs' key containing RNN input array
                (batch_size, n_timesteps, 3) with [target_x, target_y, go_cue]
        targets: Target positions for loss calculation (batch_size, n_timesteps, 4)
        init_states: Initial joint states (batch_size, 4)
        delay_go_times: Timestep when go cue appears for each trial (batch_size,)
        delay_tg_times: Timestep when target appears for each trial (batch_size,)
    """
    inputs: dict
    targets: np.ndarray
    init_states: np.ndarray
    delay_go_times: np.ndarray
    delay_tg_times: np.ndarray

    def __iter__(self):
        """Allow unpacking: inputs, targets, init_states, delay_go_times, delay_tg_times = task.generate(...)"""
        return iter([self.inputs, self.targets, self.init_states, self.delay_go_times, self.delay_tg_times])


# =============================================================================
# Task class
# =============================================================================

class ExperimentTask:
    """Generates trials for reaching task experiments.

    Supports three modes:
        - 'train': Random reaches across workspace with catch trials
        - 'train_center_out': Center-out reaches with catch trials
        - 'test_center_out': Center-out reaches without catch trials (fixed timing)

    Each trial consists of:
        - Initial hold period (target shows start position)
        - Target cue (target shows final position)
        - Go cue (signal to begin movement)
        - Catch trials (no go cue, stay at start) during training

    Args:
        effector: The biomechanical effector (provides joint2cartesian, dt, etc.)
        delay_tg: Range for target cue delay (min, max) in seconds
        delay_go: Range for go cue delay (min, max) in seconds
        run_mode: One of 'train', 'train_center_out', 'test_center_out'
        center_out_radius: Radius for center-out targets in meters
        catch_probability: Probability of catch trial during training
        joint_limits: Tuple of (min_angles, max_angles) for random reaches
        base_joint: Base joint configuration for center-out tasks
        input_noise_std: Standard deviation of noise added to inputs

    Example:
        >>> task = ExperimentTask(effector=env.effector)
        >>> task.run_mode = 'train'
        >>> inputs, targets, init_states, go_times = task.generate(batch_size=32, n_timesteps=300)
    """

    # Valid run modes
    VALID_MODES = ('train', 'train_center_out', 'test_center_out')

    def __init__(
        self,
        effector,
        delay_tg: tuple[float, float] = DEFAULT_DELAY_TARGET,
        delay_go: tuple[float, float] = DEFAULT_DELAY_GO,
        run_mode: str = 'train',
        center_out_radius: float = DEFAULT_CENTER_OUT_RADIUS,
        catch_probability: float = DEFAULT_CATCH_TRIAL_PROBABILITY,
        joint_limits: Optional[tuple[np.ndarray, np.ndarray]] = None,
        base_joint: Optional[np.ndarray] = None,
        input_noise_std: float = DEFAULT_INPUT_NOISE_STD,
    ):
        self.effector = effector
        self.dt = effector.dt
        self.delay_tg = delay_tg
        self.delay_go = delay_go
        self.run_mode = run_mode
        self.center_out_radius = center_out_radius
        self.catch_probability = catch_probability
        self.input_noise_std = input_noise_std

        # Joint limits for random reaches
        if joint_limits is not None:
            self.joint_min, self.joint_max = joint_limits
        else:
            self.joint_min = DEFAULT_JOINT_MIN
            self.joint_max = DEFAULT_JOINT_MAX

        # Base configuration for center-out
        self.base_joint = base_joint if base_joint is not None else DEFAULT_BASE_JOINT

    @property
    def run_mode(self) -> str:
        return self._run_mode

    @run_mode.setter
    def run_mode(self, value: str):
        if value not in self.VALID_MODES:
            raise ValueError(
                f"Invalid run_mode '{value}'. Must be one of: {self.VALID_MODES}"
            )
        self._run_mode = value

    def generate(
        self,
        batch_size: int,
        n_timesteps: int,
        dmin: float = 0,
        dmax: float = np.inf,
    ) -> TaskOutput:
        """Generate a batch of trials.

        Args:
            batch_size: Number of trials to generate
            n_timesteps: Number of timesteps per trial
            dmin: Minimum hand distance to target (meters, for 'train' mode)
            dmax: Maximum hand distance to target (meters, for 'train' mode)

        Returns:
            TaskOutput containing inputs, targets, init_states, and delay_go_times

        Raises:
            RuntimeError: If random target search exceeds MAX_TARGET_SEARCH_ITERATIONS
        """
        # Determine trial parameters based on mode
        if self.run_mode == 'test_center_out':
            catch_chance = 0.0
            delay_tg = (0.5, 0.5)  # Fixed timing for reproducible testing
            delay_go = (1.0, 1.0)
            init_states = np.tile(self.base_joint, (batch_size, 1))
        elif self.run_mode == 'train_center_out':
            catch_chance = self.catch_probability
            delay_tg = self.delay_tg
            delay_go = self.delay_go
            init_states = np.tile(self.base_joint, (batch_size, 1))
        else:  # 'train'
            catch_chance = self.catch_probability
            delay_tg = self.delay_tg
            delay_go = self.delay_go
            init_states = self._generate_random_init_states(batch_size)

        # Generate timing for each trial
        delay_tg_times = np.random.uniform(
            delay_tg[0] / self.dt, delay_tg[1] / self.dt, batch_size
        ).astype(int)
        delay_go_times = np.random.uniform(
            delay_go[0] / self.dt, delay_go[1] / self.dt, batch_size
        ).astype(int)

        # Determine which trials are catch trials
        is_catch = np.random.rand(batch_size) < catch_chance

        # Compute start positions in Cartesian space
        start_points = self.effector.joint2cartesian(
            th.tensor(init_states)
        ).detach().cpu().numpy()

        # Compute target positions
        if self.run_mode in ('test_center_out', 'train_center_out'):
            final_targets = self._generate_center_out_targets(batch_size, start_points)
        else:
            final_targets = self._generate_random_targets(
                batch_size, start_points, dmin, dmax
            )

        # Build input and target arrays (vectorized)
        inputs, targets = self._build_trial_arrays(
            batch_size, n_timesteps, start_points, final_targets,
            delay_tg_times, delay_go_times, is_catch
        )

        # Add noise to inputs
        noise = np.random.normal(
            loc=0.0, scale=self.input_noise_std,
            size=(batch_size, n_timesteps, 3)
        )
        inputs += noise

        return TaskOutput(
            inputs={"inputs": inputs},
            targets=targets,
            init_states=init_states,
            delay_go_times=delay_go_times,
            delay_tg_times=delay_tg_times
        )

    def _generate_random_init_states(self, batch_size: int) -> np.ndarray:
        """Generate random initial joint states within configured limits."""
        rnd = np.random.rand(batch_size, 2)
        pos = (self.joint_max - self.joint_min) * rnd + self.joint_min
        vel = np.zeros((batch_size, 2))
        return np.hstack([pos, vel]).astype(np.float32)

    def _generate_center_out_targets(
        self, batch_size: int, start_points: np.ndarray
    ) -> np.ndarray:
        """Generate targets arranged in a circle around start position."""
        angles = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
        offsets = self.center_out_radius * np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(batch_size), np.zeros(batch_size)
        ])
        return start_points + offsets

    def _generate_random_targets(
        self,
        batch_size: int,
        start_points: np.ndarray,
        dmin: float,
        dmax: float
    ) -> np.ndarray:
        """Generate random targets within distance constraints.

        Each target must satisfy:
            - Hand distance from start is in [dmin, dmax]
            - Joint angles are within configured limits
            - Hand y-position > 0 (not behind shoulder)
        """
        final_targets = np.zeros((batch_size, 4), dtype=np.float32)

        for i in range(batch_size):
            iterations = 0
            found = False

            while not found:
                iterations += 1
                if iterations > MAX_TARGET_SEARCH_ITERATIONS:
                    raise RuntimeError(
                        f"Could not find valid target after {MAX_TARGET_SEARCH_ITERATIONS} "
                        f"iterations. Try relaxing constraints (dmin={dmin}, dmax={dmax}, "
                        f"joint_limits={self.joint_min}-{self.joint_max})."
                    )

                # Sample random joint state
                tg_state_tensor = self.effector.draw_random_uniform_states(1)
                tg_state = tg_state_tensor.detach().cpu().numpy()[0]
                tg_hand = self.effector.joint2cartesian(tg_state_tensor).detach().cpu().numpy()[0]

                # Check distance constraint
                hdist = np.linalg.norm(tg_hand[0:2] - start_points[i, 0:2])

                # Check all constraints
                distance_ok = dmin <= hdist <= dmax
                joints_ok = np.all(tg_state[0:2] > self.joint_min) and \
                           np.all(tg_state[0:2] < self.joint_max)
                position_ok = tg_hand[1] > 0  # Hand not behind shoulder

                found = distance_ok and joints_ok and position_ok

            final_targets[i, 0:2] = tg_hand[0:2]

        return final_targets

    def _build_trial_arrays(
        self,
        batch_size: int,
        n_timesteps: int,
        start_points: np.ndarray,
        final_targets: np.ndarray,
        delay_tg_times: np.ndarray,
        delay_go_times: np.ndarray,
        is_catch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build input and target arrays for all trials.

        Uses vectorized operations where possible, with a loop only for
        the variable timing boundaries.
        """
        # Initialize arrays
        inputs = np.zeros((batch_size, n_timesteps, 3), dtype=np.float32)
        targets = np.zeros((batch_size, n_timesteps, start_points.shape[1]), dtype=np.float32)

        # Create time index array for vectorized comparisons
        time_idx = np.arange(n_timesteps)

        for i in range(batch_size):
            tg_time = delay_tg_times[i]
            go_time = delay_go_times[i]

            # Input: target position (start until tg_time, then final)
            inputs[i, :tg_time, 0:2] = start_points[i, 0:2]
            inputs[i, tg_time:, 0:2] = final_targets[i, 0:2]

            # Input: go cue (0 until go_time, then 1 for non-catch trials)
            # (inputs already initialized to 0)

            # Target for loss: stay at start until go_time
            targets[i, :go_time, :] = start_points[i]

            if is_catch[i]:
                # Catch trial: stay at start, no go cue
                targets[i, go_time:, :] = start_points[i]
            else:
                # Movement trial: go cue = 1, target = final position
                inputs[i, go_time:, 2] = 1.0
                targets[i, go_time:, :] = final_targets[i]

        return inputs, targets

    def __repr__(self) -> str:
        return (
            f"ExperimentTask(mode='{self.run_mode}', "
            f"delay_tg={self.delay_tg}, delay_go={self.delay_go}, "
            f"catch_prob={self.catch_probability})"
        )
