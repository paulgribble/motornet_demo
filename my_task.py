"""Task for reaching experiments: generates batches of trials for training and testing."""

import numpy as np
import torch as th

MAX_TARGET_SEARCH_ITERATIONS = 1000  # safety limit when drawing a random target


class ExperimentTask:
    """Generates batches of reaching trials.

    A trial: hold at the start position -> the target appears -> the go cue comes on -> reach.
    On catch trials (training only) the go cue never comes and the hand should stay at the start.

    run_mode selects the kind of trial:
        'train'             random reaches across the workspace, with catch trials
        'train_center_out'  center-out reaches, with catch trials
        'test_center_out'   center-out reaches, no catch trials, fixed timing

    Args:
        effector: the arm (provides dt, joint2cartesian, random joint states)
        delay_tg, delay_go: (min, max) time in seconds at which the target / go cue appears
        center_out_radius: reach distance for center-out trials (m)
        catch_probability: proportion of catch trials during training
        joint_limits: (min, max) shoulder and elbow angles for random reaches (rad)
        base_joint: start posture for center-out reaches [shoulder, elbow, shoulder_vel, elbow_vel]
        input_noise_std: standard deviation of the noise added to the network inputs
    """

    VALID_MODES = ('train', 'train_center_out', 'test_center_out')

    # Number of task inputs to the RNN at each timestep: [target_x, target_y, go_cue]
    n_inputs = 3

    def __init__(self, effector, delay_tg=(0.3, 0.8), delay_go=(0.8, 1.3), run_mode='train',
                 center_out_radius=0.10, catch_probability=0.25, joint_limits=None,
                 base_joint=None, input_noise_std=1e-3):
        self.effector = effector
        self.dt = effector.dt
        self.delay_tg = delay_tg
        self.delay_go = delay_go
        self.run_mode = run_mode
        self.center_out_radius = center_out_radius
        self.catch_probability = catch_probability
        self.input_noise_std = input_noise_std
        if joint_limits is None:
            joint_limits = (np.array([20, 20]) * np.pi / 180, np.array([110, 110]) * np.pi / 180)
        self.joint_min, self.joint_max = joint_limits
        if base_joint is None:
            base_joint = np.deg2rad([50.0, 90.0, 0.0, 0.0]).astype(np.float32)
        self.base_joint = base_joint

    @property
    def run_mode(self):
        return self._run_mode

    @run_mode.setter
    def run_mode(self, value):
        if value not in self.VALID_MODES:
            raise ValueError(f"Invalid run_mode '{value}'. Must be one of: {self.VALID_MODES}")
        self._run_mode = value

    def generate(self, batch_size, n_timesteps, dmin=0, dmax=np.inf):
        """Generate a batch of trials.

        dmin, dmax: allowed reach distance (m) for the random targets of 'train' mode.

        Returns:
            inputs:         (batch, time, 3) network inputs [target_x, target_y, go_cue]
            targets:        (batch, time, 4) desired hand state [x, y, vx, vy], for the loss
            init_states:    (batch, 4) initial joint state
            delay_go_times: (batch,) timestep of the go cue
            delay_tg_times: (batch,) timestep at which the target appears
        """
        is_test = self.run_mode == 'test_center_out'
        is_center_out = self.run_mode != 'train'

        catch_chance = 0.0 if is_test else self.catch_probability
        delay_tg = (0.5, 0.5) if is_test else self.delay_tg  # fixed timing for reproducible testing
        delay_go = (1.0, 1.0) if is_test else self.delay_go
        if is_center_out:
            init_states = np.tile(self.base_joint, (batch_size, 1))
        else:
            init_states = self._random_init_states(batch_size)

        # Timing (in timesteps) and catch trials
        delay_tg_times = np.random.uniform(delay_tg[0] / self.dt, delay_tg[1] / self.dt, batch_size).astype(int)
        delay_go_times = np.random.uniform(delay_go[0] / self.dt, delay_go[1] / self.dt, batch_size).astype(int)
        is_catch = np.random.rand(batch_size) < catch_chance

        # Start and target hand positions
        start_points = self.effector.joint2cartesian(th.tensor(init_states)).detach().cpu().numpy()
        if is_center_out:
            final_targets = self._center_out_targets(batch_size, start_points)
        else:
            final_targets = self._random_targets(batch_size, start_points, dmin, dmax)

        inputs = np.zeros((batch_size, n_timesteps, self.n_inputs), dtype=np.float32)
        targets = np.zeros((batch_size, n_timesteps, start_points.shape[1]), dtype=np.float32)
        for i in range(batch_size):
            tg_time, go_time = delay_tg_times[i], delay_go_times[i]

            # Input: the start position until the target appears, then the target position
            inputs[i, :tg_time, 0:2] = start_points[i, 0:2]
            inputs[i, tg_time:, 0:2] = final_targets[i, 0:2]

            # Loss target: stay at the start; on movement trials the go cue comes on
            # at go_time and from then the hand should be at the target
            targets[i] = start_points[i]
            if not is_catch[i]:
                inputs[i, go_time:, 2] = 1.0
                targets[i, go_time:, :] = final_targets[i]

        inputs += np.random.normal(loc=0.0, scale=self.input_noise_std, size=inputs.shape)

        return inputs, targets, init_states, delay_go_times, delay_tg_times

    def _random_init_states(self, batch_size):
        """Random initial joint angles within the joint limits, at rest."""
        pos = (self.joint_max - self.joint_min) * np.random.rand(batch_size, 2) + self.joint_min
        vel = np.zeros((batch_size, 2))
        return np.hstack([pos, vel]).astype(np.float32)

    def _center_out_targets(self, batch_size, start_points):
        """Targets equally spaced on a circle around the start position."""
        angles = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
        offsets = self.center_out_radius * np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(batch_size), np.zeros(batch_size)
        ])
        return start_points + offsets

    def _random_targets(self, batch_size, start_points, dmin, dmax):
        """Random targets: within the joint limits, between dmin and dmax from the start
        position, and in front of the shoulder (y > 0)."""
        final_targets = np.zeros((batch_size, 4), dtype=np.float32)

        for i in range(batch_size):
            for _ in range(MAX_TARGET_SEARCH_ITERATIONS):
                tg_state_tensor = self.effector.draw_random_uniform_states(1)
                tg_state = tg_state_tensor.detach().cpu().numpy()[0]

                # Check joint limits first (cheap), before computing the hand position
                if not (np.all(tg_state[0:2] > self.joint_min) and
                        np.all(tg_state[0:2] < self.joint_max)):
                    continue

                tg_hand = self.effector.joint2cartesian(tg_state_tensor).detach().cpu().numpy()[0]
                hdist = np.linalg.norm(tg_hand[0:2] - start_points[i, 0:2])
                if dmin <= hdist <= dmax and tg_hand[1] > 0:
                    break
            else:
                raise RuntimeError(
                    f"Could not find valid target after {MAX_TARGET_SEARCH_ITERATIONS} "
                    f"iterations. Try relaxing constraints (dmin={dmin}, dmax={dmax}, "
                    f"joint_limits={self.joint_min}-{self.joint_max})."
                )

            final_targets[i, 0:2] = tg_hand[0:2]

        return final_targets
