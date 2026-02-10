import torch as th


def calculate_loss_michaels(episode_data):
    # from Michaels et al. (2025) Nature https://doi.org/10.1038/s41586-025-09690-9
    losses = {
        'position'         : 1e+3 * th.mean(th.abs(episode_data['xy'][:,:,:2] - episode_data['targets'][:,:,:2])),
        'speed'            : 2e+2 * th.mean(th.square(episode_data['xy'][:,:,2:])),
        'jerk'             : 1e+6 * th.mean(th.square(th.diff(episode_data['xy'][:,:,2:],n=2,dim=1))),
        'muscle'           : 1e+0 * th.mean(episode_data['force']),
        'muscle_derivative': 0e+0 * th.mean(th.square(th.diff(episode_data['force'],n=1,dim=1))),
        'hidden'           : 1e-1 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e+4 * th.mean(th.square(th.diff(episode_data['hidden'],n=2,dim=1))) # spectral
    }
    losses['total'] = losses['position'] + losses['speed'] + losses['jerk'] + \
                      losses['muscle']   + losses['muscle_derivative'] + \
                      losses['hidden']   + losses['hidden_derivative']
    return losses


def calculate_loss_kashefi(episode_data):
    # from Kashefi et al. (2025) bioRxiv https://doi.org/10.1101/2025.09.04.674069
    losses = {
        'position'         : 1e+0 * th.mean(th.abs(episode_data['xy'][:,:,:2] - episode_data['targets'][:,:,:2])),
        'speed'            : 1e-3 * th.mean(th.square(episode_data['xy'][:,:,2:])),
        'jerk'             : 1e-4 * th.mean(th.square(th.diff(episode_data['xy'][:,:,2:],n=2,dim=1))),
        'muscle'           : 1e-4 * th.mean(episode_data['force']),
        'muscle_derivative': 1e-4 * th.mean(th.square(th.diff(episode_data['force'],n=1,dim=1))),
        'hidden'           : 1e-2 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e-1 * th.mean(th.square(th.diff(episode_data['hidden'],n=2,dim=1))) # spectral
    }
    losses['total'] = losses['position'] + losses['speed'] + losses['jerk'] + \
                      losses['muscle']   + losses['muscle_derivative'] + \
                      losses['hidden']   + losses['hidden_derivative']
    return losses

def calculate_loss_mehrdad(episode_data, policy, env):
    # from Mehrdad Kashefi demo code 'modular_paul_minimal' (November 24, 2025)
    # Get recurrent weight for weight decay (GRU weight_hh_l0)
    recurrent_weight = None
    for name, param in policy.named_parameters():
        if 'weight_hh' in name or 'recurrent' in name.lower():
            recurrent_weight = param
            break
    if recurrent_weight is None:
        recurrent_weight = th.tensor(0.0)  # Fallback if not found

    losses = {
        'pos'              : 1e+00 * th.mean(th.sum(th.abs(episode_data['xy'][:,:,:2] - episode_data['targets'][:,:,:2]), dim=-1)),
        'act'              : 0e+00 * th.mean(th.sum(th.pow(episode_data['muscle'], 2), dim=-1)),
        'force'            : 1e-04 * th.mean(th.sum(episode_data['force'], dim=-1)),
        'force_diff'       : 1e-04 * th.mean(th.sum(th.pow(th.diff(episode_data['force'], dim=1), 2), dim=-1)),
        'hdn'              : 1e-02 * th.mean(th.sum(th.pow(episode_data['hidden'], 2), dim=-1)),
        'hdn_diff'         : 1e-01 * th.mean(th.sum(th.pow(th.diff(episode_data['hidden'], dim=1), 2), dim=-1)),
        'weight_decay'     : 1e-07 * th.norm(recurrent_weight, p=2),
        'speed'            : 0e+00 * th.mean(th.sum(th.pow(episode_data['xy'][:,:,2:], 2), dim=-1)),
        'hdn_jerk'         : 1e-10 * th.mean(th.sum(th.square(th.diff(episode_data['hidden'], 3, dim=1) / th.pow(th.tensor(env.effector.dt), 3)), dim=-1)),
    }
    losses['total'] = losses['pos'] + losses['act'] + losses['force'] + \
                      losses['force_diff'] + losses['hdn'] + \
                      losses['hdn_diff'] + losses['weight_decay'] + \
                      losses['speed'] + losses['hdn_jerk']
    return losses

def calculate_loss_modular(episode_data, policy, env):
    """
    Loss function for modular reaching networks (PMd, M1, S1, SC).

    Combines the Michaels et al. (2025) structure — which produces preparatory
    dynamics — with weight decay for modular network stability.

    Key design choices:
      - Position error at 1e+3 creates strong pressure to reach accurately,
        incentivizing feedforward planning during the delay period.
      - Speed penalty at 2e+2 rewards precise, well-timed stops at the target.
      - Hidden JERK (3rd derivative) at 1e+4 encourages smooth, sustained neural
        dynamics — exactly the signature of preparatory trajectories — while
        preventing abrupt transitions. This replaces hdn_diff (1st derivative),
        which was suppressing all delay-period dynamics.
      - Hidden magnitude at 1e-1 prevents saturation but does NOT suppress
        dynamics (penalizes magnitude, not change).
      - Weight decay at 1e-7 prevents sparse inter-module connection weights
        from growing disproportionately large.

    Args:
        episode_data: dict with keys 'xy', 'hidden', 'muscle', 'targets',
                      'actions', 'joint', 'delay_go_times', etc.
        policy: the ModularPolicyGRU (needed for weight decay)
        env: the ExperimentEnv (needed for dt)

    Returns:
        dict of loss components, each a scalar tensor. 'total' is the sum.
    """
    import torch as th

    dt = env.effector.dt

    # Extract episode data
    xy = episode_data['xy']            # (batch, time, 4) — x, y, vx, vy
    hidden = episode_data['hidden']    # (batch, time, hidden_dim)
    muscle = episode_data['muscle']    # (batch, time, n_muscles)
    targets = episode_data['targets']  # (batch, time, 2) — target x, y
    actions = episode_data['actions']  # (batch, time, n_muscles)

    hand_pos = xy[:, :, :2]           # (batch, time, 2)
    hand_vel = xy[:, :, 2:]           # (batch, time, 2)

    # --- Loss weights ---
    w_position    = 1e+3
    w_speed       = 2e+2
    w_jerk        = 1e+6
    w_muscle      = 1e+0
    w_hidden      = 1e-1
    w_hidden_jerk = 1e+4
    w_weight_decay = 1e-7

    # --- Position error ---
    # Squared distance from hand to target, averaged over batch and time
    pos_error = th.sum((hand_pos - targets) ** 2, dim=-1)  # (batch, time)
    loss_position = th.mean(pos_error)

    # --- Speed penalty ---
    # Encourages the hand to stop at the target (not overshoot/oscillate)
    speed_sq = th.sum(hand_vel ** 2, dim=-1)  # (batch, time)
    loss_speed = th.mean(speed_sq)

    # --- Hand jerk ---
    # 3rd temporal derivative of hand position → smoothness of movement
    # diff once = velocity, twice = acceleration, three times = jerk
    hand_jerk = th.diff(hand_pos, n=3, dim=1) / (dt ** 3)
    loss_jerk = th.mean(hand_jerk ** 2)

    # --- Muscle force ---
    # Penalizes total muscle activation (metabolic cost)
    loss_muscle = th.mean(muscle ** 2)

    # --- Hidden state magnitude ---
    # Light regularization to prevent saturation; does NOT suppress dynamics
    loss_hidden = th.mean(hidden ** 2)

    # --- Hidden state jerk (3rd derivative) ---
    # Encourages SMOOTH neural dynamics: slow preparatory trajectories are
    # cheap (low jerk), abrupt transitions are expensive (high jerk).
    # This is the critical replacement for hdn_diff — it permits delay-period
    # dynamics while penalizing non-smooth transitions.
    hidden_jerk = th.diff(hidden, n=3, dim=1) / (dt ** 3)
    loss_hidden_jerk = th.mean(hidden_jerk ** 2)

    # --- Weight decay ---
    # L2 penalty on recurrent weights. Important for modular networks where
    # sparse inter-module connections can grow disproportionately large.
    loss_weight_decay = th.tensor(0.0, device=hidden.device)
    for name, param in policy.named_parameters():
        if 'hh' in name or 'recurrent' in name or 'weight' in name:
            loss_weight_decay = loss_weight_decay + th.sum(param ** 2)

    # --- Total ---
    total = (
        w_position     * loss_position
        + w_speed      * loss_speed
        + w_jerk       * loss_jerk
        + w_muscle     * loss_muscle
        + w_hidden     * loss_hidden
        + w_hidden_jerk * loss_hidden_jerk
        + w_weight_decay * loss_weight_decay
    )

    return {
        'total':        total,
        'position':     loss_position.detach(),
        'speed':        loss_speed.detach(),
        'jerk':         loss_jerk.detach(),
        'muscle':       loss_muscle.detach(),
        'hidden':       loss_hidden.detach(),
        'hidden_jerk':  loss_hidden_jerk.detach(),
        'weight_decay': loss_weight_decay.detach(),
    }

