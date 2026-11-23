"""Running simulations, and saving / loading models."""

import os
import json
import torch as th
import motornet as mn
from my_env import ExperimentEnv
from my_task import ExperimentTask
from my_policy import PolicyGRU

# Rotates a velocity vector by 90 degrees clockwise (curl force field)
_CURL_CW = th.tensor([[0., -1.], [1., 0.]])


def applied_load(endpoint_vel, k, mode='CW'):
    """Velocity-dependent curl force field of strength k ('CW' clockwise or 'CCW' counter-clockwise).

    endpoint_vel and the returned force are (batch, 2).
    """
    return k * endpoint_vel @ (_CURL_CW if mode == 'CW' else _CURL_CW.T)


def run_episode(env, task, policy, batch_size, n_t, device, k=0, dmax=0.30):
    """Simulate one batch of trials from start to finish.

    Args:
        env, task, policy: the environment, task and network
        batch_size: number of trials
        n_t: number of timesteps of task input to generate (the episode itself ends
             when env.max_ep_duration is reached)
        k: force field strength (0 = no force field)
        dmax: maximum reach distance for random reaches (m)

    Returns a dict of (batch, time, ...) tensors: 'xy' (hand position and velocity),
    'joint', 'hidden', 'actions', 'muscle' (activation), 'force', 'targets', 'inp'
    (task inputs), plus 'l1', 'l2' (arm segment lengths), 'dt', 'delay_go_times'
    and 'delay_tg_times'.
    """
    inputs, targets, init_states, delay_go_times, delay_tg_times = task.generate(batch_size, n_t, dmax=dmax)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs, device=device, dtype=th.float)
    init_states = th.tensor(init_states, device=device, dtype=th.float)
    h = policy.init_hidden(batch_size)
    obs, info = env.reset(options={'batch_size': batch_size, 'joint_state': init_states})
    terminated = False

    # Force field is off during the hold period (go cue ~0) and on once the go cue is given (~1).
    # The go cue input is 0 or 1 plus a little noise, so threshold halfway.
    force_mask = (inp[:, :, 2] > 0.5).float().unsqueeze(2)

    # Per-timestep storage (stacked along the time axis at the end)
    log = {key: [] for key in ('xy', 'hidden', 'actions', 'muscle', 'force', 'joint')}
    steps = []

    while not terminated:
        t_step = int(env.elapsed / env.dt)

        # the network sees the sensory feedback followed by the task inputs
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        if k == 0:
            obs, _, terminated, _, info = env.step(action=action)
        else:
            force_field = applied_load(endpoint_vel=info['states']['cartesian'][:, 2:], k=k, mode='CW')
            obs, _, terminated, _, info = env.step(action=action, endpoint_load=force_field * force_mask[:, t_step])

        steps.append(t_step)
        log['xy'].append(info['states']['cartesian'])
        log['hidden'].append(h)
        log['actions'].append(action)
        log['muscle'].append(info['states']['muscle'][:, 0, :])   # activation
        log['force'].append(info['states']['muscle'][:, -1, :])
        log['joint'].append(info['states']['joint'])

    # PolicyGRU: h is (1, batch, hidden); ModularPolicyGRU: h is (batch, hidden)
    hidden = th.stack(log['hidden'], dim=-2)
    if hidden.dim() == 4:
        hidden = hidden.squeeze(0)

    return {
        'xy': th.stack(log['xy'], dim=1),
        'hidden': hidden,
        'actions': th.stack(log['actions'], dim=1),
        'muscle': th.stack(log['muscle'], dim=1),
        'force': th.stack(log['force'], dim=1),
        'targets': targets[:, steps, :],
        'inp': inp[:, steps, :],
        'joint': th.stack(log['joint'], dim=1),
        'l1': env.skeleton.l1,
        'l2': env.skeleton.l2,
        'dt': env.dt,
        'delay_go_times': delay_go_times,
        'delay_tg_times': delay_tg_times
    }


def model_path(model_name, suffix):
    """Path of a model's file on disk, e.g. model_path('demo1', 'cfg.json') -> 'demo1/demo1_cfg.json'."""
    return os.path.join(model_name, f"{model_name}_{suffix}")


def save_model(env, policy, losses, model_name):
    """Save a PolicyGRU model: weights, loss history and environment configuration."""
    th.save(policy.state_dict(), model_path(model_name, "weights.pkl"))
    with open(model_path(model_name, "losses.json"), 'w') as f:
        json.dump(losses, f)
    with open(model_path(model_name, "cfg.json"), 'w') as f:
        json.dump(env.get_save_config(), f)
    for suffix in ("weights.pkl", "losses.json", "cfg.json"):
        print(f"saved {model_path(model_name, suffix)}")


def load_model(cfg_file, weight_file):
    """Rebuild a model saved by save_model. Returns (env, task, policy, device)."""
    device = th.device("cpu")
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    # The saved config stores delays in timesteps and noise as one value per element
    dt = cfg['effector']['dt']
    muscle = getattr(mn.muscle, cfg['effector']['muscle']['name'])()
    env = ExperimentEnv(
        effector=mn.effector.RigidTendonArm26(muscle=muscle, timestep=dt),
        max_ep_duration=cfg['max_ep_duration'],
        proprioception_delay=cfg['proprioception_delay'] * cfg['dt'],
        vision_delay=cfg['vision_delay'] * cfg['dt'],
        proprioception_noise=cfg['proprioception_noise'][0],
        vision_noise=cfg['vision_noise'][0],
        action_noise=cfg['action_noise'][0],
    )
    task = ExperimentTask(effector=env.effector)

    weights = th.load(weight_file, weights_only=True)
    n_hidden = weights['gru.weight_hh_l0'].shape[1]
    policy = PolicyGRU(env.observation_space.shape[0] + task.n_inputs, n_hidden, env.n_muscles, device=device)
    policy.load_state_dict(weights)
    return env, task, policy, device
