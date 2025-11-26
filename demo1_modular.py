import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' # Set this too just in case
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'

import torch as th
th.set_num_threads(1)                       # intra-op
th.set_num_interop_threads(1)               # inter-op

import numpy as np
import matplotlib.pyplot as plt
import motornet as mn
from tqdm import tqdm
import pickle

from my_policy import ModularPolicyGRU       # the modular RNN
from my_loss   import calculate_loss_michaels as calculate_loss # the loss function
from my_env    import ExperimentEnv  # the environment
from my_task   import ExperimentTask # the task
from my_utils  import run_episode    # run a batch of simulations
from my_utils  import plot_losses    # for plotting loss history
from my_utils  import plot_handpaths # for plotting hand paths
from my_utils  import plot_signals_modular   # for plotting inputs and outputs per trial
from my_utils  import save_model_modular     # for saving model config, weights, losses to disk
from my_utils  import load_model_modular     # for loading model config and weights from disk

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
                    proprioception_delay=0.02, vision_delay=0.07,
                    proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4
                    )

obs, info = env.reset()
n_t = int(ep_dur / env.effector.dt)

# define the experimental task
task = ExperimentTask(effector=env.effector)
inputs, targets, init_states, _ = task.generate(1, n_t) # get sample inputs so we know how many for policy
n_task_inputs = inputs['inputs'].shape[2]
task_dim = np.arange(inputs['inputs'].shape[-1])
vision_dim = np.arange(env.get_vision().shape[1]) + task_dim[-1] + 1
proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1

# define the modular RNN
policy = ModularPolicyGRU(input_size  = env.observation_space.shape[0] + n_task_inputs,
                          module_size = [256, 256, 32],
                          output_size = env.n_muscles,
                          vision_dim  = vision_dim,
                          proprio_dim = proprio_dim,
                          task_dim    = task_dim,
                          vision_mask = [0.2, 0.0, 0.0],
                          proprio_mask = [0.0, 0.0, 0.5],
                          task_mask = [0.2, 0.02, 0.0],
                          connectivity_mask = np.array([[1. , 0.2 , 0.02], [0.2 , 1.  , 0.2 ], [0.  , 0.2 , 1.  ]]),
                          output_mask = [0.0, 0.0, 0.5],
                          connectivity_delay = np.zeros((3,3)),
                          spectral_scaling = 1.1,
                          device = device,
                          activation = 'tanh')

# Set the optimizer and constraints
optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)


# --------------------------------------------------
# TRAINING 
# --------------------------------------------------

n_batch       = 10000
interval      =   100   # for intermediate plots
batch_size    =    64
FF_k          =     0   # force-field strength
save_name     = "demo1_modular" # name to use to save model, plots, etc

# Create a directory to store output plots and files if it doesn't already exist
if not os.path.exists(save_name):
        print(f"creating directory {save_name}/")
        os.mkdir(save_name)

# Pre-allocate numpy arrays instead of lists for better memory efficiency
loss_history = {
    "total"             : np.zeros(n_batch, dtype=np.float32),
    "position"          : np.zeros(n_batch, dtype=np.float32),
    "speed"             : np.zeros(n_batch, dtype=np.float32),
    "jerk"              : np.zeros(n_batch, dtype=np.float32),
    "muscle"            : np.zeros(n_batch, dtype=np.float32),
    "muscle_derivative" : np.zeros(n_batch, dtype=np.float32),
    "hidden"            : np.zeros(n_batch, dtype=np.float32),
    "hidden_derivative" : np.zeros(n_batch, dtype=np.float32),
}

# Optimize loss keys for faster iteration
loss_keys = ["total", "position", "speed", "jerk", 
             "muscle", "muscle_derivative", "hidden", "hidden_derivative"]

# main training loop
print(f"main training loop")
print(f"simulating {ep_dur} second movements using {n_t} time points")

for i in tqdm(
    iterable = range(n_batch), 
    desc     = f"training {n_batch} batches of {batch_size}", 
    unit     = "batch", 
    total    = n_batch, 
):
    
    task.run_mode = 'train'                                      # random reaches in workspace
    episode_data = run_episode(env, task, policy, batch_size, n_t, device, k=FF_k) # run the batch forwards
    loss = calculate_loss(episode_data)                  # calculate loss
    loss['total'].backward()                                     # propagate loss backwards to compute gradients
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1) # so that gradients don't get out of hand
    optimizer.step()                                             # adjust network weights
    optimizer.zero_grad(set_to_none=True)                        # zero out the gradients
    
    # Track losses
    with th.no_grad():  # Don't need gradients for loss tracking
        for key in loss_keys:
            loss_history[key][i] = loss[key].item()
    
    # plot some things every once in a while
    if (i>0) and (i % interval == 0):
        # Convert to lists the portions we need for plotting
        current_history = {key: loss_history[key][:i+1].tolist() for key in loss_keys}
        plot_losses(loss_history=current_history, fname=os.path.join(save_name, f"{save_name}_losses.png"))
        plot_handpaths(episode_data=episode_data, fname=os.path.join(save_name, f"{save_name}_handpaths.png"), figtitle=f"batch {i:04d} (n={batch_size})")
        for j in range(4):  # plot first 4 trials of batch
            plot_signals_modular(episode_data=episode_data, fname=os.path.join(save_name,f"{save_name}_signals_{j}.png"), figtitle=f"batch {i:04d} (n={batch_size})", trial=j)

# Convert back to lists
for key in loss_keys:
    loss_history[key] = loss_history[key].tolist()

# save the model to disk
save_model_modular(env, policy, loss_history, save_name)


# --------------------------------------------------
# TESTING
# --------------------------------------------------

# LOAD A MODEL AND TEST IT ON CENTER-OUT REACHES

save_name = "demo1_modular"

n_tg     = 8    # number of targets for center-out task
sim_time = 3.00 # simulation time (seconds)
FF_k     = 0    # FF strength

print(f"loading model {save_name}")
env,task,policy,device = load_model_modular(os.path.join(save_name,f"{save_name}_cfg.json"), os.path.join(save_name,f"{save_name}_weights.pkl"))

n_t = int(sim_time / env.dt)           # number of simulation steps
task.run_mode = 'test_center_out'      # center-out reaches

print(f"simulating {task.run_mode}")
episode_data = run_episode(env, task, policy, n_tg, n_t, device, k=FF_k) # run the batch forwards

# Plot results
plot_handpaths(episode_data=episode_data, fname=os.path.join(save_name,f"{save_name}_handpaths_test.png"))
for i in range(n_tg):
    plot_signals_modular(episode_data=episode_data, fname=os.path.join(save_name,f"{save_name}_signals_test_{i}.png"), figtitle=f"trial {i}", trial=i)

# save episide data to a .pkl file on disk
with open(os.path.join(save_name,f"{save_name}_episode_data.pkl"), "wb") as f:
    pickle.dump(episode_data, f)

