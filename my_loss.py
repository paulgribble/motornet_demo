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

def michaels_modular_loss(episode_data):
    losses = {
        'position'         : 1e+3 * th.mean(th.sum(th.abs(episode_data['xy'][:, :, 0:2] - episode_data['targets'][:, :, :2]), dim=-1)),
        'muscle'           : 1e-1 * th.mean(th.sum(episode_data['force'], dim=-1)),
        'hidden'           : 1e-2 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e+3 * th.mean(th.sum(th.square(th.diff(episode_data['hidden'], 2, dim=1)), dim=-1)),
        'jerk'             : 1e+5 * th.mean(th.sum(th.square(th.diff(episode_data['xy'][:, :, 2:], 2, dim=1)), dim=-1)),
    }
    losses['total'] = losses['position'] + losses['muscle'] + losses['hidden'] + losses['hidden_derivative'] + losses['jerk']
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

