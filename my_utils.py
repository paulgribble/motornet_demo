import os
import torch as th
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import motornet as mn
from my_env import ExperimentEnv
from my_task import ExperimentTask
from my_policy import Policy
from my_policy import ModularPolicyGRU

# Pre-computed curl matrices for force field (avoid recreating every timestep)
_CURL_CW = th.tensor([[0., -1.], [1., 0.]])
_CURL_CCW = th.tensor([[0., 1.], [-1., 0.]])


def applied_load(endpoint_vel, k, mode='CW'):
    """Apply a velocity-dependent curl force field.

    Args:
        endpoint_vel: Endpoint velocity tensor (batch, 2)
        k: Force field strength
        mode: 'CW' for clockwise, 'CCW' for counter-clockwise

    Returns:
        Force field tensor (batch, 2)
    """
    curl_matrix = _CURL_CW if mode == 'CW' else _CURL_CCW
    return k * endpoint_vel @ curl_matrix


def run_episode(env, task, policy, batch_size, n_t, device, k=0, **kwargs):
    """Run a single episode (batch of trials).

    Args:
        env: The environment
        task: The task generator
        policy: The policy network
        batch_size: Number of trials in the batch
        n_t: Number of timesteps
        device: Torch device
        k: Force field strength (default: 0)

    Returns:
        Dictionary containing episode data (xy, hidden, actions, muscle, force, etc.)
    """
    inputs, targets, init_states, delay_go_times = task.generate(batch_size, n_t, dmax=0.30)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs['inputs'], device=device, dtype=th.float)
    init_states = th.tensor(init_states, device=device, dtype=th.float)
    h = policy.init_hidden(batch_size)
    obs, info = env.reset(options={'batch_size': batch_size, 'joint_state': init_states})
    terminated = False

    # Initialize storage lists
    xy = []
    all_actions = []
    all_muscle = []
    all_hidden = []
    all_force = []
    all_targets = []
    all_inp = []
    all_joint = []

    while not terminated:
        t_step = int(env.elapsed / env.dt)
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        # Compute endpoint load (force field)
        # Force field is masked during hold period (when go cue is ~0)
        force_mask = (inp[:, t_step, 2].abs() < 1e-3).float().unsqueeze(1)
        force_field = applied_load(endpoint_vel=info['states']['cartesian'][:, 2:], k=k, mode='CW')
        masked_force_field = force_field * force_mask

        obs, _, terminated, _, info = env.step(action=action, endpoint_load=masked_force_field)

        # Store data - handle hidden state shape difference between simple and modular policies
        # Simple GRU: h has shape (1, batch, hidden) -> squeeze to (batch, hidden)
        # Modular GRU: h has shape (batch, hidden) -> use as-is
        h_to_store = h.squeeze(0) if h.dim() == 3 else h

        xy.append(info['states']['cartesian'][:, None, :])
        all_actions.append(action[:, None, :])
        all_muscle.append(info['states']['muscle'][:, 0, None, :])
        all_force.append(info['states']['muscle'][:, -1, None, :])
        all_hidden.append(h_to_store[:, None, :])
        all_targets.append(targets[:, t_step:t_step+1, :])
        all_inp.append(inp[:, t_step:t_step+1, :])
        all_joint.append(info['states']['joint'][:, None, :])

    return {
        'xy': th.cat(xy, dim=1),
        'hidden': th.cat(all_hidden, dim=1),
        'actions': th.cat(all_actions, dim=1),
        'muscle': th.cat(all_muscle, dim=1),
        'force': th.cat(all_force, dim=1),
        'targets': th.cat(all_targets, dim=1),
        'inp': th.cat(all_inp, dim=1),
        'joint': th.cat(all_joint, dim=1),
        'l1': env.skeleton.l1,
        'l2': env.skeleton.l2,
        'dt': env.dt,
        'delay_go_times': delay_go_times
    }


def plot_losses(loss_history, fname=""):
    """Plot training loss history.

    Args:
        loss_history: Dictionary mapping loss names to lists of values
        fname: If provided, save figure to this path; otherwise display

    Returns:
        (fig, ax) tuple if fname is empty, otherwise None
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))

    for loss_name in loss_history.keys():
        ax[0].plot(loss_history[loss_name], alpha=0.5)
        ax[1].semilogy(loss_history[loss_name], alpha=0.5)

    leg0 = ax[0].legend(loss_history.keys(), loc='upper right')
    for line in leg0.get_lines():
        line.set_linewidth(2)

    ax[1].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')

    # Set y-limit based on initial total loss (if available)
    if 'total' in loss_history and len(loss_history['total']) > 0:
        ax[0].set_ylim([0, loss_history['total'][0]])

    fig.tight_layout()

    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()
        return fig, ax


def plot_handpaths(episode_data, fname="", figtitle=""):
    """Plot hand trajectories for all trials in an episode.

    Args:
        episode_data: Dictionary from run_episode()
        fname: If provided, save figure to this path; otherwise display
        figtitle: Title for the figure

    Returns:
        (fig, ax) tuple if fname is empty, otherwise None
    """
    xy = episode_data['xy'].detach()
    tg = episode_data['targets'].detach()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot targets: final (blue squares), initial (red dots), and trajectories (dashed)
    ax.plot(tg[:, -1, 0], tg[:, -1, 1], 'bs', alpha=0.5)
    ax.plot(tg[:, 0, 0], tg[:, 0, 1], 'r.', alpha=0.5)
    ax.plot(tg[:, :, 0].T, tg[:, :, 1].T, '--', lw=0.5, alpha=0.5)

    # Reset color cycle and plot hand paths
    ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    ax.plot(xy[:, :, 0].T, xy[:, :, 1].T, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Hand Paths')
    ax.axis('equal')
    fig.suptitle(figtitle, fontsize=14)
    fig.tight_layout()

    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()
        return fig, ax


def plot_signals(episode_data, fname="", figtitle="", trial=0, coord="xy"):
    """Plot detailed signals for a single trial.

    This function works for both simple and modular policies.

    Args:
        episode_data: Dictionary from run_episode()
        fname: If provided, save figure to this path; otherwise display
        figtitle: Title for the figure
        trial: Which trial to plot (0-indexed)
        coord: 'xy' for Cartesian coordinates, 'joint' for joint angles

    Returns:
        (fig, ax) tuple if fname is empty, otherwise None
    """
    # Extract and prepare data based on coordinate system
    if coord == "xy":
        pos = episode_data['xy'].detach()[:, :, 0:2]
        vel = episode_data['xy'].detach()[:, :, 2:]
        tg = episode_data['targets'].detach()
        inp = episode_data['inp'].detach()
        pos_labels = ('X (m)', 'Y (m)')
        vel_label = 'XY VEL (m/s)'
    else:  # joint coordinates
        pos = episode_data['joint'].detach()[:, :, 0:2] * 180 / np.pi
        vel = episode_data['joint'].detach()[:, :, 2:] * 180 / np.pi
        tg = xy_to_joints(episode_data['targets'][:, :, 0:2].detach().numpy(),
                         episode_data['l1'], episode_data['l2']) * 180 / np.pi
        inp_np = episode_data['inp'].detach().numpy().copy()
        inp_np[:, :, 0:2] = xy_to_joints(inp_np[:, :, 0:2],
                                          episode_data['l1'], episode_data['l2']) * 180 / np.pi
        inp = inp_np
        pos_labels = ('SHOULDER (deg)', 'ELBOW (deg)')
        vel_label = 'JOINT VEL (deg/s)'

    hidden = episode_data['hidden'].detach()
    activation = episode_data['muscle'].detach()
    n_timesteps = pos.shape[1]

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(6, 13), constrained_layout=True)
    gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[1, 2, 2, 2, 4, 4])
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    # Plot go cue
    axes[0].plot(inp[trial, :, 2], '-')
    axes[0].set_ylabel('GO CUE [0,1]')
    axes[0].set_ylim([-0.01, 1.01])

    # Plot position X (or shoulder)
    axes[1].plot(inp[trial, :, 0], ':')
    axes[1].plot(pos[trial, :, 0], '-')
    axes[1].plot(tg[trial, :, 0], '--')
    axes[1].set_ylabel(pos_labels[0])

    # Plot position Y (or elbow)
    axes[2].plot(inp[trial, :, 1], ':')
    axes[2].plot(pos[trial, :, 1], '-')
    axes[2].plot(tg[trial, :, 1], '--')
    axes[2].set_ylabel(pos_labels[1])

    # Plot velocity
    axes[3].plot(vel[trial, :, :], '-')
    axes[3].set_ylabel(vel_label)

    # Plot hidden activity
    axes[4].plot(hidden[trial, :, :], '-', alpha=0.25)
    axes[4].set_ylabel('GRU HIDDEN')

    # Plot muscle activation
    axes[5].plot(activation[trial, :, :], '-')
    axes[5].set_ylabel('MUSCLE ACTIVATION [0,1]')
    axes[5].set_xlabel('TIME (steps)')

    # Style all axes
    for i, ax in enumerate(axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, n_timesteps])
        ax.set_xticks(np.arange(0, n_timesteps, 50))
        if i < 4:
            ax.set_xticks([])
            ax.tick_params(axis='x', length=0)
            ax.spines['bottom'].set_visible(False)

    fig.suptitle(figtitle, fontsize=14)

    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()
        return fig, axes


# Alias for backwards compatibility
plot_signals_modular = plot_signals


def xy_to_joints(xy, l1, l2):
    """Convert Cartesian coordinates to joint angles using inverse kinematics.

    Uses vectorized operations for efficiency.

    Args:
        xy: Cartesian coordinates, shape (2,), (N, 2), or (batch, N, 2)
        l1: Length of first arm segment
        l2: Length of second arm segment

    Returns:
        Joint angles in radians, same shape as input
    """
    xy = np.asarray(xy)
    original_shape = xy.shape

    # Handle different input dimensions
    if xy.ndim == 1:
        xy = xy.reshape(1, 2)
    elif xy.ndim == 3:
        # Reshape (batch, N, 2) to (batch*N, 2)
        batch, n, _ = xy.shape
        xy = xy.reshape(-1, 2)

    # Vectorized inverse kinematics
    x, y = xy[:, 0], xy[:, 1]

    # Compute elbow angle (a1) using law of cosines
    cos_a1 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_a1 = np.clip(cos_a1, -1.0, 1.0)
    a1 = np.arccos(cos_a1)

    # Compute shoulder angle (a0)
    # Use atan2 to handle all quadrants correctly and avoid division by zero
    a0 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(a1), l1 + l2 * np.cos(a1))

    # Normalize shoulder angle to [0, pi] range
    a0 = np.where(a0 < 0, np.pi + a0, a0)
    a0 = np.where(a0 > np.pi, a0 - np.pi, a0)

    joints = np.stack([a0, a1], axis=-1)

    # Reshape back to original shape
    if len(original_shape) == 1:
        return joints.squeeze(0)
    elif len(original_shape) == 3:
        return joints.reshape(original_shape)
    return joints


def save_model(env, policy, losses, model_name, quiet=False):
    """Save a trained model to disk.

    Args:
        env: The environment
        policy: The policy network
        losses: Dictionary of training losses
        model_name: Name/directory for the model
        quiet: If True, suppress print statements
    """
    weight_file = os.path.join(model_name, f"{model_name}_weights.pkl")
    losses_file = os.path.join(model_name, f"{model_name}_losses.json")
    cfg_file = os.path.join(model_name, f"{model_name}_cfg.json")

    # Save model weights
    th.save(policy.state_dict(), weight_file)

    # Save training history
    with open(losses_file, 'w') as f:
        json.dump(losses, f)

    # Save environment configuration
    cfg = env.get_save_config()

    # Add module_size if this is a modular policy
    if hasattr(policy, 'module_size'):
        cfg['module_size'] = policy.module_size

    with open(cfg_file, 'w') as f:
        json.dump(cfg, f)

    if not quiet:
        print(f"saved {weight_file}")
        print(f"saved {losses_file}")
        print(f"saved {cfg_file}")


# Alias for backwards compatibility
save_model_modular = save_model


def load_model(cfg_file, weight_file):
    """Load a previously trained model from disk.

    Automatically detects whether the model is simple or modular.

    Args:
        cfg_file: Path to the configuration JSON file
        weight_file: Path to the weights file

    Returns:
        Tuple of (env, task, policy, device)
    """
    device = th.device("cpu")

    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    # Recreate the environment
    dt = cfg['effector']['dt']
    muscle_name = cfg['effector']['muscle']['name']
    muscle = getattr(mn.muscle, muscle_name)()
    effector = mn.effector.RigidTendonArm26(muscle=muscle, timestep=dt)

    proprioception_delay = cfg['proprioception_delay'] * cfg['dt']
    vision_delay = cfg['vision_delay'] * cfg['dt']
    action_noise = cfg['action_noise'][0]
    proprioception_noise = cfg['proprioception_noise'][0]
    vision_noise = cfg['vision_noise'][0]
    max_ep_duration = cfg['max_ep_duration']

    env = ExperimentEnv(
        effector=effector,
        max_ep_duration=max_ep_duration,
        proprioception_delay=proprioception_delay,
        vision_delay=vision_delay,
        proprioception_noise=proprioception_noise,
        vision_noise=vision_noise,
        action_noise=action_noise
    )

    # Load weights to determine architecture
    weights = th.load(weight_file, weights_only=True)

    # Create task to get input dimensions
    task = ExperimentTask(effector=env.effector)
    inputs, _, _, _ = task.generate(1, 1)
    n_task_inputs = inputs['inputs'].shape[2]
    input_size = env.observation_space.shape[0] + n_task_inputs

    # Detect if this is a modular model
    is_modular = 'module_size' in cfg

    if is_modular:
        task_dim = np.arange(inputs['inputs'].shape[-1])
        vision_dim = np.arange(env.get_vision().shape[1]) + task_dim[-1] + 1
        proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1

        n_modules = len(cfg['module_size'])

        # Read masks from config if present, otherwise use defaults matching module count
        if n_modules == 3:
            default_vision = [0.2, 0.0, 0.0]
            default_proprio = [0.0, 0.0, 0.5]
            default_task = [0.2, 0.02, 0.0]
            default_conn = [[1., 0.2, 0.02], [0.2, 1., 0.2], [0., 0.2, 1.]]
            default_output = [0.0, 0.0, 0.5]
        else:
            default_vision = [0.2, 0.0, 0.0, 0.0]
            default_proprio = [0.0, 0.0, 0.5, 0.3]
            default_task = [0.2, 0.02, 0.0, 0.0]
            default_conn = [
                [1.0, 0.1, 0.05, 0.0],
                [0.2, 1.0, 0.2, 0.0],
                [0.05, 0.1, 1.0, 0.1],
                [0.0, 0.2, 0.05, 1.0]
            ]
            default_output = [0.0, 0.0, 0.0, 0.5]

        vision_mask = cfg.get('vision_mask', default_vision)
        proprio_mask = cfg.get('proprio_mask', default_proprio)
        task_mask = cfg.get('task_mask', default_task)
        connectivity_mask = cfg.get('connectivity_mask', default_conn)
        output_mask = cfg.get('output_mask', default_output)
        spectral_scaling = cfg.get('spectral_scaling', 1.1)

        policy = ModularPolicyGRU(
            input_size=input_size,
            module_size=cfg['module_size'],
            output_size=env.n_muscles,
            vision_dim=vision_dim,
            proprio_dim=proprio_dim,
            task_dim=task_dim,
            vision_mask=vision_mask,
            proprio_mask=proprio_mask,
            task_mask=task_mask,
            connectivity_mask=np.array(connectivity_mask),
            output_mask=output_mask,
            connectivity_delay=np.zeros((n_modules, n_modules)),
            spectral_scaling=spectral_scaling,
            device=device,
            activation='tanh'
        )
    else:
        # Simple GRU - infer hidden size from weights
        n_hidden = weights['gru.weight_ih_l0'].shape[0] // 3
        policy = Policy(input_size, n_hidden, env.n_muscles, device=device)

    policy.load_state_dict(weights)
    return env, task, policy, device


# Alias for backwards compatibility
load_model_modular = load_model
