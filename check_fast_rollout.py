"""Check that fast_rollout.py computes the same thing as the standard simulation (my_utils.run_episode).

Run this after changing anything in motornet/, my_policy.py, my_env.py or fast_rollout.py:

    uv run python check_fast_rollout.py              # eager fast path (about 1 minute)
    uv run python check_fast_rollout.py --compiled   # also the compiled preset used for training (slower)

Part 1 switches all noise off and feeds both implementations the same trials and weights: the
simulated trajectories, the loss and the gradients must then agree to float32 round-off.
Part 2 switches the noise on and checks that it has the same size and structure in both.
"""

import os
import sys
import copy
import tempfile

import numpy as np
import torch as th

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(tempfile.mkdtemp(prefix='check_fast_'))   # the test models are saved here, not in the repo

from reaching_model import ReachingModel
from my_utils import run_episode
from my_loss import michaels_modular_loss
from fast_rollout import run_episode_fast, get_fast_rollout, FAST_PRESET

N_T = 300
TOL_TRAJECTORY = 2e-5   # max abs difference (positions in m, activity, activation)
TOL_LOSS = 1e-5         # relative
TOL_GRADIENT = 2e-4     # relative (L2 norm over all parameters)

ARCHITECTURE = dict(
    module_names=["premotor", "motor", "spinal"], module_sizes=[128, 128, 32],
    vision_mask=[1.0, 1.0, 0.0], proprio_mask=[0.0, 0.0, 1.0], task_mask=[1.0, 0.0, 0.0],
    connectivity_mask=[[0.7, 0.2, 0.0], [0.5, 0.7, 0.1], [0.0, 0.5, 0.7]], output_mask=[0.0, 0.0, 1.0],
)


def loss_and_gradients(model, episode_data):
    loss = michaels_modular_loss(episode_data)['total']
    loss.backward()
    grads = th.cat([p.grad.flatten() for p in model.policy.parameters() if p.grad is not None])
    model.policy.zero_grad(set_to_none=True)
    return loss.item(), grads


def check_without_noise(variants, train_batches):
    model = ReachingModel.create("check", proprioception_noise=0.0, vision_noise=0.0, action_noise=0.0, **ARCHITECTURE)
    model._save_training_plots = lambda *args, **kwargs: None
    if train_batches:   # train a little (with noise) so that the arm actually moves
        model.train(n_batches=train_batches, batch_size=32, plot_interval=0, quiet=True)
    model.policy.hidden_noise = 0.0
    env, task, policy = model.env, model.task, model.policy

    ok = True
    for mode, batch_size, k in [('train', 32, 0), ('train_center_out', 32, 0), ('test_center_out', 8, 0),
                                ('train', 32, 8.0)]:
        task.run_mode = mode
        batch = task.generate(batch_size, N_T, dmax=0.30)
        generate, task.generate = task.generate, lambda *args, **kwargs: copy.deepcopy(batch)
        ref = run_episode(env, task, policy, batch_size, N_T, model.device, k=k)
        ref_loss, ref_grads = loss_and_gradients(model, ref)
        for name, options in variants.items():
            out = run_episode_fast(env, task, policy, batch_size, N_T, model.device, k=k, **options)
            loss, grads = loss_and_gradients(model, out)
            assert set(out) == set(ref)
            d_traj = max((out[key] - ref[key]).abs().max().item()
                         for key in ('xy', 'joint', 'hidden', 'actions', 'muscle'))
            d_force = ((out['force'] - ref['force']).abs().max() / ref['force'].abs().max()).item()
            d_loss = abs(loss - ref_loss) / abs(ref_loss)
            d_grad = ((grads - ref_grads).norm() / ref_grads.norm()).item()
            passed = (d_traj < TOL_TRAJECTORY and d_force < TOL_TRAJECTORY and d_loss < TOL_LOSS
                      and d_grad < TOL_GRADIENT)
            ok &= passed
            print(f"{'ok  ' if passed else 'FAIL'} {name:8s} {mode:17s} ff={k:<4} trajectories {d_traj:.1e}  "
                  f"force (rel) {d_force:.1e}  loss (rel) {d_loss:.1e}  gradients (rel) {d_grad:.1e}")
        task.generate = generate
    return ok


def check_noise():
    """With the arm held still (no muscle drive would be ideal, but the untrained network barely
    moves it in the first steps), compare what the network is fed in the two implementations."""
    model = ReachingModel.create("check_noise", **ARCHITECTURE)
    env, task, policy = model.env, model.task, model.policy
    task.run_mode = 'test_center_out'
    batch_size, n_steps = 256, 40
    batch = task.generate(batch_size, n_steps)
    task.generate = lambda *args, **kwargs: copy.deepcopy(batch)
    n_vision = env.get_vision().shape[1]

    # standard code: record the observations passed to the network
    seen = []
    hook = policy.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    with th.no_grad():
        run_episode(env, task, policy, batch_size, n_steps, model.device)
    hook.remove()
    stock = th.stack(seen)[:, :, :-task.n_inputs]                      # (time, batch, vision + proprio)

    fast = get_fast_rollout(env, policy)
    fast.keep_obs = True
    with th.no_grad():
        run_episode_fast(env, task, policy, batch_size, n_steps, model.device)
    fast_obs = fast.last_obs[:, :, :stock.shape[2]]

    ok = True
    for name, obs in (('standard', stock), ('fast', fast_obs)):
        vision = obs[:env.vision_delay, :, :n_vision]                  # the hand has not been seen to move yet
        noise = vision - vision.mean(dim=(0, 1), keepdim=True)         # all trials start at the same posture
        std = noise.std().item()
        across_channels = np.corrcoef(noise[:, :, 0].flatten(), noise[:, :, 1].flatten())[0, 1]
        across_steps = np.corrcoef(noise[:-1].flatten(), noise[1:].flatten())[0, 1]
        proprio = obs[:, :, n_vision:]
        p_noise = (proprio[1:] - proprio[:-1])[:2]                     # differences of successive samples
        p_channels = np.corrcoef(p_noise[:, :, 0].flatten(), p_noise[:, :, 1].flatten())[0, 1]
        passed = (abs(std / env.vision_noise[0] - 1) < 0.05 and abs(across_channels) < 0.05
                  and abs(across_steps) < 0.05 and abs(p_channels) < 0.1)
        ok &= passed
        print(f"{'ok  ' if passed else 'FAIL'} {name:8s} vision noise std {std:.2e} (expected {env.vision_noise[0]:.0e})  "
              f"correlation: x-y {across_channels:+.3f}, successive steps {across_steps:+.3f}, "
              f"proprioception channels {p_channels:+.3f}")
    return ok


if __name__ == "__main__":
    variants = {'eager': dict()}
    if '--compiled' in sys.argv:
        variants['compiled'] = FAST_PRESET
    th.manual_seed(0)
    np.random.seed(0)
    print("Noise off, untrained network:")
    ok = check_without_noise(variants, train_batches=0)
    print("Noise off, after 150 batches of training:")
    ok &= check_without_noise(variants, train_batches=150)
    print("Noise on:")
    ok &= check_noise()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
