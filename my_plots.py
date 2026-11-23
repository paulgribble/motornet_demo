"""Plots of training progress and of simulated trials.

Each function saves the figure to fname if one is given; otherwise it shows the
figure and returns (fig, axes).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _save_or_show(fig, axes, fname):
    if fname:
        fig.savefig(fname)
        plt.close(fig)
        return None
    plt.show()
    return fig, axes


def plot_losses(loss_history, fname=""):
    """Plot each loss term over training batches, on linear (top) and log (bottom) axes.

    loss_history: dict mapping loss names to lists of values
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))

    for values in loss_history.values():
        ax[0].plot(values, alpha=0.5)
        ax[1].semilogy(values, alpha=0.5)

    leg0 = ax[0].legend(loss_history.keys(), loc='upper right')
    for line in leg0.get_lines():
        line.set_linewidth(2)

    ax[1].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')

    # Scale the linear plot to the initial total loss
    if loss_history.get('total'):
        ax[0].set_ylim([0, loss_history['total'][0]])

    fig.tight_layout()
    return _save_or_show(fig, ax, fname)


def plot_handpaths(episode_data, fname="", figtitle=""):
    """Plot the hand paths of all trials in an episode (the dict returned by run_episode)."""
    xy = episode_data['xy'].detach()
    tg = episode_data['targets'].detach()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Targets: final (blue squares), initial (red dots), and target trajectories (dashed)
    ax.plot(tg[:, -1, 0], tg[:, -1, 1], 'bs', alpha=0.5)
    ax.plot(tg[:, 0, 0], tg[:, 0, 1], 'r.', alpha=0.5)
    ax.plot(tg[:, :, 0].T, tg[:, :, 1].T, '--', lw=0.5, alpha=0.5)

    # Hand paths, restarting the color cycle so they match their targets
    ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    ax.plot(xy[:, :, 0].T, xy[:, :, 1].T, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Hand Paths')
    ax.axis('equal')
    fig.suptitle(figtitle, fontsize=14)
    fig.tight_layout()
    return _save_or_show(fig, ax, fname)


def plot_signals(episode_data, fname="", figtitle="", trial=0, coord="xy"):
    """Plot the signals of one trial over time: go cue, position, velocity, hidden
    activity and muscle activation.

    coord: 'xy' plots hand position and velocity, 'joint' plots joint angles and velocities
    """
    if coord == "xy":
        pos = episode_data['xy'].detach()[:, :, 0:2]
        vel = episode_data['xy'].detach()[:, :, 2:]
        tg = episode_data['targets'].detach()
        inp = episode_data['inp'].detach()
        pos_labels = ('X (m)', 'Y (m)')
        vel_label = 'XY VEL (m/s)'
    else:
        l1, l2 = episode_data['l1'], episode_data['l2']
        pos = episode_data['joint'].detach()[:, :, 0:2] * 180 / np.pi
        vel = episode_data['joint'].detach()[:, :, 2:] * 180 / np.pi
        tg = xy_to_joints(episode_data['targets'][:, :, 0:2].detach().numpy(), l1, l2) * 180 / np.pi
        inp = episode_data['inp'].detach().numpy().copy()
        inp[:, :, 0:2] = xy_to_joints(inp[:, :, 0:2], l1, l2) * 180 / np.pi
        pos_labels = ('SHOULDER (deg)', 'ELBOW (deg)')
        vel_label = 'JOINT VEL (deg/s)'

    hidden = episode_data['hidden'].detach()
    activation = episode_data['muscle'].detach()
    n_timesteps = pos.shape[1]

    fig = plt.figure(figsize=(6, 13), constrained_layout=True)
    gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[1, 2, 2, 2, 4, 4])
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    axes[0].plot(inp[trial, :, 2], '-')
    axes[0].set_ylabel('GO CUE [0,1]')
    axes[0].set_ylim([-0.01, 1.01])

    # Position: target input (dotted), actual (solid), loss target (dashed)
    for d in (0, 1):
        axes[1 + d].plot(inp[trial, :, d], ':')
        axes[1 + d].plot(pos[trial, :, d], '-')
        axes[1 + d].plot(tg[trial, :, d], '--')
        axes[1 + d].set_ylabel(pos_labels[d])

    axes[3].plot(vel[trial, :, :], '-')
    axes[3].set_ylabel(vel_label)

    axes[4].plot(hidden[trial, :, :], '-', alpha=0.25)
    axes[4].set_ylabel('GRU HIDDEN')

    axes[5].plot(activation[trial, :, :], '-')
    axes[5].set_ylabel('MUSCLE ACTIVATION [0,1]')
    axes[5].set_xlabel('TIME (steps)')

    # Only the bottom two panels show a time axis
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
    return _save_or_show(fig, axes, fname)


def xy_to_joints(xy, l1, l2):
    """Inverse kinematics of the two-joint arm.

    xy: hand positions, shape (..., 2); l1, l2: upper arm and forearm lengths.
    Returns [shoulder, elbow] angles in radians, same shape as xy.
    """
    xy = np.asarray(xy)
    x, y = xy[..., 0], xy[..., 1]

    # Elbow angle from the law of cosines
    cos_a1 = np.clip((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2), -1.0, 1.0)
    a1 = np.arccos(cos_a1)

    # Shoulder angle, normalized to [0, pi]
    a0 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(a1), l1 + l2 * np.cos(a1))
    a0 = np.where(a0 < 0, np.pi + a0, a0)
    a0 = np.where(a0 > np.pi, a0 - np.pi, a0)

    return np.stack([a0, a1], axis=-1)
