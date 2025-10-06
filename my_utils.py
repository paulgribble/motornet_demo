import os
import torch as th
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import motornet as mn
from my_env    import ExperimentEnv  # the environment
from my_task   import ExperimentTask # a task
from my_policy import Policy         # the RNN

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
    inputs, targets, init_states, delay_go_times = task.generate(batch_size, n_t, dmax=0.30) # max 30 cm target distance
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
        'xy'             : th.cat(xy, dim=1),
        'hidden'         : th.cat(all_hidden, dim=1),
        'actions'        : th.cat(all_actions, dim=1),
        'muscle'         : th.cat(all_muscle, dim=1),
        'force'          : th.cat(all_force, dim=1),
        'targets'        : th.cat(all_targets, dim=1),
        'inp'            : th.cat(all_inp, dim=1),
        'joint'          : th.cat(all_joint, dim=1),
        'l1'             : env.skeleton.l1,
        'l2'             : env.skeleton.l2,
        'dt'             : env.dt,
        'delay_go_times' : delay_go_times
    }

def plot_losses(loss_history, fname=""):
    fig,ax = plt.subplots(2,1, figsize=(8,10))
    for l in loss_history.keys():
        ax[0].plot(loss_history[l], alpha=0.5)
        ax[1].semilogy(loss_history[l], alpha=0.5)
    leg0 = ax[0].legend(loss_history.keys(), loc='upper right')
    leg1 = ax[1].legend(loss_history.keys(), loc='lower center')
    for line in leg0.get_lines():
        line.set_linewidth(2)  # change this number as desired
    for line in leg1.get_lines():
        line.set_linewidth(2)  # change this number as desired
    ax[1].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[0].set_ylim([0,loss_history['total'][0]])
    fig.tight_layout()
    if not fname=="":
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()
        return fig,ax

def plot_handpaths(episode_data, fname="", figtitle=""):
    xy = episode_data['xy'].detach()
    tg = episode_data['targets'].detach()
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(tg[:,-1,0],tg[:,-1,1],'bs', alpha=0.5)
    ax.plot(tg[:,0,0],tg[:,0,1],'r.', alpha=0.5)
    ax.plot(tg[:,:,0].T,tg[:,:,1].T,'--',lw=0.5, alpha=0.5)
    ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    ax.plot(xy[:,:,0].T,xy[:,:,1].T, alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Hand Paths')
    ax.axis('equal')
    fig.suptitle(figtitle, fontsize=14)
    fig.tight_layout()
    if not fname=="":
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()
        return fig,ax

def plot_signals(episode_data, fname="", figtitle="", trial=0, coord="xy"):
    if (coord=="xy"):
        xy = episode_data['xy'].detach()[:,:,0:2]
        vel = episode_data['xy'].detach()[:,:,2:]
        tg = episode_data['targets'].detach()
        inp = episode_data['inp'].detach()
    elif (coord=="joint"):
        xy = episode_data['joint'].detach()[:,:,0:2] * 180/np.pi
        vel = episode_data['joint'].detach()[:,:,2:] * 180/np.pi
        tg = xy_to_joints(episode_data['targets'][:,:,0:2].detach(), episode_data['l1'], episode_data['l2']) * 180 / np.pi
        inp = episode_data['inp'].detach().numpy()
        inp_tmp = inp.copy()
        inp[:,:,0:2] = xy_to_joints(inp_tmp[:,:,0:2], episode_data['l1'], episode_data['l2']) * 180 / np.pi
    hidden = episode_data['hidden'].detach()
    hidden = np.transpose(np.squeeze(hidden, axis=0), (1,0,2))
    activation = episode_data['muscle'].detach()
    fig = plt.figure(figsize=(6, 13), constrained_layout=True)
    gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[1, 2, 2, 2, 4, 4])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])
    ax5 = fig.add_subplot(gs[5])
    ax = [ax0, ax1, ax2, ax3, ax4, ax5]
    ax[0].plot(inp[trial,:,2],'-')
    ax[0].set_ylabel('GO CUE [0,1]')
    ax[0].set_ylim([-0.01,1.01])
    ax[1].plot(inp[trial,:,0],':')
    ax[1].plot(xy[trial,:,0],'-')
    ax[1].plot(tg[trial,:,0],'--')
    if (coord=="xy"):
        ax[1].set_ylabel('X (m)')
    elif (coord=="joint"):
        ax[1].set_ylabel('SHOULDER (deg)')
    ax[2].plot(inp[trial,:,1],':')
    ax[2].plot(xy[trial,:,1],'-')
    ax[2].plot(tg[trial,:,1],'--')
    if (coord=="xy"):
        ax[2].set_ylabel('Y (m)')
    elif (coord=="joint"):
        ax[2].set_ylabel('ELBOW (deg)')
    ax[3].plot(vel[trial,:,:],'-')
    if (coord=="xy"):
        ax[3].set_ylabel('XY VEL (m/s)')
    elif (coord=="joint"):
        ax[3].set_ylabel('JOINT VEL (deg/s)')
    ax[4].plot(hidden[trial,:,:],'-', alpha=0.25)
    ax[4].set_ylabel('GRU HIDDEN')
    ax[5].plot(activation[trial,:,:],'-')
    ax[5].set_ylabel('MUSCLE ACTIVATION [0,1]')
    ax[5].set_xlabel('TIME (steps)')
    for i in range(6):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlim([0,np.shape(xy)[1]])
        ax[i].set_xticks(np.arange(0, np.shape(xy)[1], 50))
        if i<4:
            ax[i].set_xticks([])
            ax[i].tick_params(axis='x', length=0)
            ax[i].spines['bottom'].set_visible(False)
    fig.suptitle(figtitle, fontsize=14)
    if not fname=="":
        fig.savefig(fname)
        plt.close(fig)
    else:
        plt.show()
        return fig,ax


def xy_to_joints_helper(xy, l1, l2):
    a0,a1 = 0,0
    tmp = ((xy[0]*xy[0])+(xy[1]*xy[1])-(l1*l1)-(l2*l2))/(2*l1*l2)
    tmp = np.clip(tmp, -1.0, 1.0)
    a1 = np.acos(tmp)
    a0 = np.atan(xy[1]/xy[0]) - np.atan((l2*np.sin(a1))/(l1+(l2*np.cos(a1))))
    if a0 < 0:
        a0 = np.pi+a0
    elif a0 > np.pi:
        a0 = a0-np.pi
    return np.array([a0,a1])

def xy_to_joints(xy, l1, l2):
    if (len(np.shape(xy)) == 1):
        joints = xy_to_joints_helper(xy, l1, l2)
    elif (len(np.shape(xy)) == 2):
        r,c = np.shape(xy)
        joints = np.zeros((r,c))
        for i in range(r):
            joints[i,:] = xy_to_joints_helper(xy[i,:], l1, l2)
    elif (len(np.shape(xy)) == 3):
        z,r,c = np.shape(xy)
        joints = np.zeros((z,r,c))
        for iz in range(z):
            for i in range(r):
                joints[iz,i,:] = xy_to_joints_helper(xy[iz,i,:], l1, l2)
    return joints


def save_model(env, policy, losses, model_name, quiet=False):
    weight_file = os.path.join(model_name, model_name + "_weights.pkl")
    losses_file = os.path.join(model_name, model_name + "_losses.json")
    cfg_file    = os.path.join(model_name, model_name + "_cfg.json")

    # save model weights
    th.save(policy.state_dict(), weight_file)

    # save training history (log)
    with open(losses_file, 'w') as file:
        json.dump(losses, file)

    # save environment configuration dictionary
    cfg = env.get_save_config()
    with open(cfg_file, 'w') as file:
        json.dump(cfg, file)

    if (quiet == False):
        print(f"saved {weight_file}")
        print(f"saved {losses_file}")
        print(f"saved {cfg_file}")


def load_model(cfg_file, weight_file):
    # load a previously trained model
    device = th.device("cpu")
    cfg = json.load(open(cfg_file, 'r'))
    dt = cfg['effector']['dt']
    muscle_name = cfg['effector']['muscle']['name']
    muscle = getattr(mn.muscle, muscle_name)()
    effector = mn.effector.RigidTendonArm26(muscle=muscle, timestep=dt)
    proprioception_delay = cfg['proprioception_delay']*cfg['dt']
    vision_delay = cfg['vision_delay']*cfg['dt']
    action_noise = cfg['action_noise'][0]
    proprioception_noise = cfg['proprioception_noise'][0]
    vision_noise = cfg['vision_noise'][0]
    max_ep_duration = cfg['max_ep_duration']
    env = ExperimentEnv(effector=effector, max_ep_duration=max_ep_duration,
                        proprioception_delay=proprioception_delay, vision_delay=vision_delay,
                        proprioception_noise=proprioception_noise, vision_noise=vision_noise,
                        action_noise=action_noise
                        )
    w = th.load(weight_file, weights_only=True)
    n_hidden = int(w['gru.weight_ih_l0'].shape[0]/3)
    task = ExperimentTask(effector=env.effector)
    inputs, _, _, _ = task.generate(1, 1) # just to get input shape
    policy = Policy(env.observation_space.shape[0] + inputs['inputs'].shape[2], n_hidden, env.n_muscles, device=device)
    policy.load_state_dict(w)
    return env, task, policy, device
