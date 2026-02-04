"""
Custom environment for reaching experiments.

This module defines ExperimentEnv, a simplified environment that extends
motornet's base Environment class with a streamlined observation space.
"""

import motornet as mn
import torch as th
import numpy as np
from typing import Any, Optional

__all__ = ["ExperimentEnv"]


class ExperimentEnv(mn.environment.Environment):
    """Environment for arm reaching experiments.

    This environment extends the base motornet Environment with a simplified
    observation space that includes only vision and proprioception (no goal).

    Key differences from parent class:
        - Observations contain [vision, proprioception] only (no goal tensor)
        - Does not use action_frame_stacking in observations
        - Proprioception is normalized muscle length and velocity

    The observation vector structure is:
        [fingertip_x, fingertip_y, muscle_lengths..., muscle_velocities...]

    Args:
        *args: Passed to parent Environment class
        **kwargs: Passed to parent Environment class. Key parameters include:
            - effector: The biomechanical effector (e.g., RigidTendonArm26)
            - max_ep_duration: Maximum episode duration in seconds
            - proprioception_delay: Delay for proprioceptive feedback (seconds)
            - vision_delay: Delay for visual feedback (seconds)
            - proprioception_noise: Std dev of proprioceptive noise
            - vision_noise: Std dev of visual noise
            - action_noise: Std dev of motor command noise
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "ExperimentEnv"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[th.Tensor, dict]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Dictionary with optional settings:
                - batch_size (int): Number of parallel trials (default: 1)
                - joint_state (Tensor): Initial joint configuration (default: random)
                - deterministic (bool): If True, disable observation noise (default: False)

        Returns:
            obs: Initial observation tensor
            info: Dictionary containing:
                - states: Current biomechanical states
                - action: Initial action (zeros)
                - noisy action: Same as action (no noise on reset)
        """
        self._set_generator(seed)

        options = options or {}
        batch_size: int = options.get("batch_size", 1)
        joint_state: Optional[th.Tensor | np.ndarray] = options.get("joint_state", None)
        deterministic: bool = options.get("deterministic", False)

        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})
        self.elapsed = 0.0
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)

        # Initialize observation buffers with current sensory state
        # Note: get_proprioception() and get_vision() apply their own noise
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self.states,
            "action": action,
            "noisy action": action,
        }
        return obs, info

    def step(
        self,
        action: th.Tensor,
        deterministic: bool = False,
        **kwargs
    ) -> tuple[th.Tensor, None, bool, bool, dict]:
        """Execute one simulation step.

        Args:
            action: Motor command tensor of shape (batch_size, n_muscles)
            deterministic: If True, disable action and observation noise
            **kwargs: Additional arguments passed to effector.step(), e.g.:
                - endpoint_load: External forces applied at the fingertip

        Returns:
            obs: Observation tensor after the step
            reward: Always None (not used in this differentiable environment)
            terminated: True if episode duration reached
            truncated: Always False
            info: Dictionary containing:
                - states: Current biomechanical states
                - action: The commanded action (before noise)
                - noisy action: The actual action executed (after noise)
        """
        self.elapsed += self.dt

        if not deterministic:
            noisy_action = self.apply_noise(action, noise=self.action_noise)
        else:
            noisy_action = action

        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(action=noisy_action)
        reward = None
        truncated = False
        terminated = bool(self.elapsed >= self.max_ep_duration)
        info = {
            "states": self.states,
            "action": action,
            "noisy action": noisy_action,
        }
        return obs, reward, terminated, truncated, info

    def get_proprioception(self) -> th.Tensor:
        """Get proprioceptive feedback (muscle lengths and velocities).

        Returns normalized muscle fiber lengths and velocities, with noise applied.

        Note: Noise is applied here, so even with deterministic=True in step(),
        the values stored in the observation buffer will have noise. The
        deterministic flag only affects the final observation noise in get_obs().

        Returns:
            Tensor of shape (batch_size, 2 * n_muscles) containing:
                [normalized_lengths..., normalized_velocities...]
        """
        mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
        mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
        prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
        return self.apply_noise(prop, self.proprioception_noise)

    def get_vision(self) -> th.Tensor:
        """Get visual feedback (fingertip position).

        Returns the fingertip position with noise applied.

        Note: Noise is applied here, so even with deterministic=True in step(),
        the values stored in the observation buffer will have noise. The
        deterministic flag only affects the final observation noise in get_obs().

        Returns:
            Tensor of shape (batch_size, 2) containing [x, y] fingertip position
        """
        vis = self.states["fingertip"]
        return self.apply_noise(vis, self.vision_noise)

    def get_obs(
        self,
        action: Optional[th.Tensor] = None,
        deterministic: bool = False
    ) -> th.Tensor:
        """Construct the observation vector from buffered sensory feedback.

        Unlike the parent class, this observation does NOT include:
            - Goal position (task provides this separately)
            - Past actions (action_frame_stacking not used)

        The observation uses delayed sensory feedback based on the configured
        proprioception_delay and vision_delay.

        Args:
            action: Current action to store in buffer (optional)
            deterministic: If True, skip final observation noise

        Returns:
            Tensor of shape (batch_size, obs_dim) containing:
                [vision (oldest), proprioception (oldest)]
        """
        self.update_obs_buffer(action=action)

        obs_as_list = [
            self.obs_buffer["vision"][0],           # oldest element (delayed)
            self.obs_buffer["proprioception"][0],   # oldest element (delayed)
        ]
        obs = th.cat(obs_as_list, dim=-1)

        if not deterministic:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs
