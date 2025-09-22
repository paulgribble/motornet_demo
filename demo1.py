import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' # Set this too just in case

import torch as th
th.set_num_threads(1)           # intra-op
th.set_num_interop_threads(1)   # inter-op
th._dynamo.config.cache_size_limit = 64

import numpy as np
import matplotlib.pyplot as plt
import motornet as mn
from tqdm import tqdm
import json

from my_policy import Policy       # the RNN
from my_loss import calculate_loss # the loss function
from my_env import ExperimentEnv   # the environment
from my_task import ExperimentTask # the task
from my_utils import run_episode   # run a batch of simulations
from my_utils import plot_losses   # for plotting loss history

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)


# use the cpu not the gpu
device = th.device("cpu")

# define a two-joint planar arm using a Hill-type muscle model as described in:
#   Kistemaker, Wong & Gribble (2010) J. Neurophysiol. 104(6):2985-94
effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

# initialize the experimental environment
ep_dur = 3.00 # duration of each simulation (seconds)
env = ExperimentEnv(effector=effector, max_ep_duration=ep_dur,
                    proprioception_delay=0.01, vision_delay=0.07,
                    proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4
                    )

obs, info = env.reset()
n_t = int(ep_dur / env.effector.dt)

# define the experimental task
task = ExperimentTask(effector=env.effector)
inputs, targets, init_states = task.generate(1, n_t)

# define the RNN
n_units = 256
policy = Policy(env.observation_space.shape[0] + inputs['inputs'].shape[2], n_units, env.n_muscles, device=device)

# define the learning rule for updating RNN weights
optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)


# Training loop

n_batch       = 5000
batch_size    =   64
task.run_mode = 'train' # random reaching across the workspace
FF_k          = 0.0     # force-field strength

loss_history = {
    "total": [],
    "position": [],
    "speed": [],
    "jerk": [],
    "muscle": [],
    "muscle_derivative": [],
    "hidden": [],
    "hidden_derivative": [],
}

print(f"simulating {ep_dur} second movements using {n_t} time points")
for i in tqdm(
    iterable = range(n_batch), 
    desc     = f"training {n_batch} batches of {batch_size}", 
    unit     = "batch", 
    total    = n_batch, 
):
    inputs, targets, init_states = task.generate(batch_size, n_t)
    episode_data = run_episode(env, task, policy, batch_size, n_t, device, k=FF_k)
    loss = calculate_loss(episode_data)
    loss['total'].backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)  # important to make sure gradients don't get crazy
    optimizer.step()
    optimizer.zero_grad()
    loss_history["total"].append(loss["total"].item())
    loss_history["position"].append(loss["position"].item())
    loss_history["speed"].append(loss["speed"].item())
    loss_history["jerk"].append(loss["jerk"].item())
    loss_history["muscle"].append(loss["muscle"].item())
    loss_history["muscle_derivative"].append(loss["muscle_derivative"].item())
    loss_history["hidden"].append(loss["hidden"].item())
    loss_history["hidden_derivative"].append(loss["hidden_derivative"].item())


with open('demo1_losses.json', 'w') as file:
    json.dump(loss_history, file)

# Test performance on the center-out task

task.run_mode = 'test_center_out'
n_tg = 8 # targets around a circle
inputs, targets, init_states = task.generate(n_tg, n_t)
episode_data = run_episode(env, task, policy, n_tg, n_t, device, k=0)

# Plot results

xy = episode_data['xy'].detach()
tg = episode_data['targets'].detach()
inp = episode_data['inp'].detach()

fig,ax = plt.subplots()
ax.plot(tg[:,0,0],tg[:,0,1],'rs')
ax.plot(tg[:,-1,0],tg[:,-1,1],'bs')
ax.plot(xy[:,:,0].T,xy[:,:,1].T)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Hand Paths')
ax.axis('equal')
fig.tight_layout()
fig.savefig("demo1_handpaths.png")
plt.close(fig)

plot_losses("demo1_losses.png", loss_history)

