import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' # Set this too just in case
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'

import torch as th
th._dynamo.reset()                          # clear compilation cache
th.set_num_threads(1)                       # intra-op
th.set_num_interop_threads(1)               # inter-op
th._dynamo.config.cache_size_limit = 64     # Smaller cache for CPU workloads

import numpy as np
import matplotlib.pyplot as plt
import motornet as mn
from tqdm import tqdm
import pickle

from my_policy import Policy         # the RNN
from my_loss   import calculate_loss # the loss function
from my_env    import ExperimentEnv  # the environment
from my_task   import ExperimentTask # the task
from my_utils  import run_episode    # run a batch of simulations
from my_utils  import plot_losses    # for plotting loss history
from my_utils  import plot_handpaths # for plotting hand paths
from my_utils  import plot_signals   # for plotting inputs and outputs per trial
from my_utils  import save_model     # for saving model config, weights, losses to disk
from my_utils  import load_model     # for loading model config and weights from disk

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)


########## DEFINE A NEW MODEL AND TRAIN IT ON RANDOM REACHES ##########

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


# Main training loop

n_batch       = 10000
interval      =   100   # for intermediate plots
batch_size    =    64
FF_k          =     0   # force-field strength

# Pre-allocate numpy arrays instead of lists for better memory efficiency
loss_history = {
    "total": np.zeros(n_batch, dtype=np.float32),
    "position": np.zeros(n_batch, dtype=np.float32),
    "speed": np.zeros(n_batch, dtype=np.float32),
    "jerk": np.zeros(n_batch, dtype=np.float32),
    "muscle": np.zeros(n_batch, dtype=np.float32),
    "muscle_derivative": np.zeros(n_batch, dtype=np.float32),
    "hidden": np.zeros(n_batch, dtype=np.float32),
    "hidden_derivative": np.zeros(n_batch, dtype=np.float32),
}

# Optimize loss keys for faster iteration
loss_keys = ["total", "position", "speed", "jerk", 
             "muscle", "muscle_derivative", "hidden", "hidden_derivative"]

print(f"simulating {ep_dur} second movements using {n_t} time points")

for i in tqdm(
    iterable = range(n_batch), 
    desc     = f"training {n_batch} batches of {batch_size}", 
    unit     = "batch", 
    total    = n_batch, 
):
    
    task.run_mode = 'train'
    episode_data = run_episode(env, task, policy, batch_size, n_t, device, k=FF_k)
    loss = calculate_loss(episode_data)
    loss['total'].backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # More memory efficient
    
    # Track ALL losses efficiently - vectorized assignment with no gradients
    with th.no_grad():  # Don't need gradients for loss tracking
        for key in loss_keys:
            loss_history[key][i] = loss[key].item()
    
    # Plotting optimizations while keeping same frequency
    if (i>0) and (i % interval == 0):
        # Convert to lists only for the portion we need for plotting
        current_history = {key: loss_history[key][:i+1].tolist() for key in loss_keys}
        plot_losses(current_history, f"demo1_losses.png")
        plot_handpaths(episode_data, "demo1_handpaths.png", figtitle=f"batch {i:04d} (n={batch_size})")
        for j in range(4):  # plot 4 example trials
            plot_signals(episode_data, f"demo1_signals_{j}.png", figtitle=f"batch {i:04d} (n={batch_size})", trial=j)

# Convert back to lists at the end for compatibility with your existing code
for key in loss_keys:
    loss_history[key] = loss_history[key].tolist()

save_model(env, policy, loss_history, "demo1")



########## LOAD A MODEL AND TEST IT ON CENTER-OUT REACHES ##########

n_tg     = 8    # number of targets for center-out task
sim_time = 3.00 # simulation time (seconds)
FF_k     = 0    # FF strength

env,task,policy,device = load_model("demo1_cfg.json", "demo1_weights.pkl")
n_t = int(sim_time / env.dt) # simulation steps
task.run_mode = 'test_center_out'
inputs, targets, init_states = task.generate(n_tg, n_t)
episode_data = run_episode(env, task, policy, n_tg, n_t, device, k=FF_k)

# Plot results

plot_handpaths(episode_data)
for i in range(n_tg):
    plot_signals(episode_data, figtitle=f"trial {i}", trial=i)

# save episide data to a .pkl file
with open("demo1_episode_data.pkl", "wb") as f:
    pickle.dump(episode_data, f)

