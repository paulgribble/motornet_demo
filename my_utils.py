import torch as th
import matplotlib.pyplot as plt
import numpy as np

# Apply a curl force field
def applied_load(endpoint_vel, k, mode = 'CW'):
    # Curved Force
    if mode == 'CW':
        curl_matrix = th.tensor([[0., -1.], [1., 0.]])   # Clockwise FF
    else:
        curl_matrix = th.tensor([[0., 1.], [-1., 0.]])   # CounterClockwise FF
    force_field = k * endpoint_vel @ curl_matrix
    return force_field

# Run a single episode
def run_episode(env, task, policy, batch_size, n_t, device, k = 0, *args, **kwargs ):
    # run a single batch
    inputs, targets, init_states = task.generate(batch_size, n_t)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs['inputs'], device=device, dtype=th.float)
    init_states = th.tensor(init_states, device=device, dtype=th.float)
    h = policy.init_hidden(batch_size)
    obs, info = env.reset(options={'batch_size': batch_size, 'joint_state': init_states})
    terminated = False

    # initialize things we want to keep track of
    xy = []
    all_actions = [] # RNN stim to muscle
    all_muscle = []  # muscle activation
    all_hidden = []  # RNN hidden activity
    all_force = []   # muscle force
    all_targets = [] # target x,y position and vel
    all_inp = []     # inputs to RNN (x,y target and go cue)
    all_joint = []   # joint angles and velocities

    while not terminated:  # will run until `max_ep_duration` is reached
        t_step = int(env.elapsed / env.dt)
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        # Compute endpoint load (force field)
        force_mask = (inp[:, t_step, 2].abs() < 1e-3).float().unsqueeze(1)
        force_field  = applied_load(endpoint_vel = info['states']['cartesian'][:, 2:], k = k, mode = 'CW')
        masked_force_field = force_field * force_mask

        obs, _, terminated, _, info = env.step(action=action, endpoint_load = masked_force_field)
        xy.append(info['states']['cartesian'][:, None, :])
        all_actions.append(action[:, None, :])
        all_muscle.append(info['states']['muscle'][:, 0, None, :])
        all_force.append(info['states']['muscle'][:, -1, None, :])
        all_hidden.append(h[:, None, :])
        all_targets.append(th.unsqueeze(targets[:, t_step, :], dim=1))
        all_inp.append(th.unsqueeze(inp[:, t_step, :], dim=1))
        all_joint.append(info['states']['joint'][:, None, :])

    return {
        'xy': th.cat(xy, dim=1),
        'hidden' : th.cat(all_hidden, dim=1),
        'actions' : th.cat(all_actions, dim=1),
        'muscle' : th.cat(all_muscle, dim=1),
        'force' : th.cat(all_force, dim=1),
        'targets' : th.cat(all_targets, dim=1),
        'inp' : th.cat(all_inp, dim=1),
        'joint' : th.cat(all_joint, dim=1)
    }

def plot_losses(fname, loss_history):
    fig,ax = plt.subplots(2,1, figsize=(8,10))
    for l in loss_history.keys():
        ax[0].plot(loss_history[l])
        ax[1].semilogy(loss_history[l])
    leg0 = ax[0].legend(loss_history.keys(), loc='upper right')
    leg1 = ax[1].legend(loss_history.keys(), loc='center right')
    for line in leg0.get_lines():
        line.set_linewidth(2)  # change this number as desired
    for line in leg1.get_lines():
        line.set_linewidth(2)  # change this number as desired
    ax[1].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

def plot_handpaths(fname, episode_data, figtitle=""):
    xy = episode_data['xy'].detach()
    tg = episode_data['targets'].detach()
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(tg[:,0,0],tg[:,0,1],'r.', alpha=0.5)
    ax.plot(tg[:,-1,0],tg[:,-1,1],'bs', alpha=0.5)
    ax.plot(tg[:,:,0].T,tg[:,:,1].T,'--',lw=0.5, alpha=0.5)
    ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    ax.plot(xy[:,:,0].T,xy[:,:,1].T, alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Hand Paths')
    ax.axis('equal')
    fig.suptitle(figtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
