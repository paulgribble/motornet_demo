"""Environment for reaching experiments."""

import torch as th
import motornet as mn


class ExperimentEnv(mn.environment.Environment):
    """A motornet Environment whose observation is only delayed, noisy vision and proprioception:

        [fingertip_x, fingertip_y, muscle lengths..., muscle velocities...]

    Unlike the parent class the observation contains no goal and no past actions;
    the task supplies the target to the network separately (see my_task.py).
    Everything else (reset, step, sensory delays and noise) is inherited from motornet.

    Useful keyword arguments: effector, max_ep_duration (s), proprioception_delay (s),
    vision_delay (s), proprioception_noise, vision_noise, action_noise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "ExperimentEnv"  # recorded in the saved config

    def get_obs(self, action=None, deterministic: bool = False) -> th.Tensor:
        self.update_obs_buffer(action=action)
        obs = th.cat([
            self.obs_buffer["vision"][0],          # oldest element = delayed feedback
            self.obs_buffer["proprioception"][0],
        ], dim=-1)
        if not deterministic:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs
