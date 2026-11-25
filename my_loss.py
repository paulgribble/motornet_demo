"""Loss functions.

Each takes the episode_data dict returned by run_episode() and returns a dict of
weighted loss terms, plus their sum under 'total' (the quantity that training minimizes).
To try a different loss, add a function here with the same form.
"""

import torch as th


def _with_total(terms):
    return {'total': sum(terms.values()), **terms}


def calculate_loss_michaels(episode_data):
    # from Michaels et al. (2025) Nature https://doi.org/10.1038/s41586-025-09690-9
    return _with_total({
        'position'         : 1e+3 * th.mean(th.abs(episode_data['xy'][:,:,:2] - episode_data['targets'][:,:,:2])),
        'speed'            : 2e+2 * th.mean(th.square(episode_data['xy'][:,:,2:])),
        'jerk'             : 1e+6 * th.mean(th.square(th.diff(episode_data['xy'][:,:,2:],n=2,dim=1))),
        'muscle'           : 1e+0 * th.mean(episode_data['force']),
        'muscle_derivative': 0e+0 * th.mean(th.square(th.diff(episode_data['force'],n=1,dim=1))),
        'hidden'           : 1e-1 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e+4 * th.mean(th.square(th.diff(episode_data['hidden'],n=2,dim=1))), # spectral
    })


def michaels_modular_loss(episode_data):
    # the loss used for the modular networks (see reaching_model.py)
    return _with_total({
        'position'         : 1e+3 * th.mean(th.sum(th.abs(episode_data['xy'][:, :, 0:2] - episode_data['targets'][:, :, :2]), dim=-1)),
        'muscle'           : 1e-1 * th.mean(th.sum(episode_data['force'], dim=-1)),
        'hidden'           : 1e-2 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e+3 * th.mean(th.sum(th.square(th.diff(episode_data['hidden'], 2, dim=1)), dim=-1)),
        'jerk'             : 1e+5 * th.mean(th.sum(th.square(th.diff(episode_data['xy'][:, :, 2:], 2, dim=1)), dim=-1)),
    })

def plg_modular_loss(episode_data):
    # the loss used for the modular networks (see reaching_model.py)
    return _with_total({
        'position'         : 1e+3 * th.mean(th.sum(th.abs(episode_data['xy'][:, :, 0:2] - episode_data['targets'][:, :, :2]), dim=-1)),
        'muscle'           : 1e-1 * th.mean(th.sum(episode_data['force'], dim=-1)),
        'hidden'           : 1e+2 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e+3 * th.mean(th.sum(th.square(th.diff(episode_data['hidden'], 2, dim=1)), dim=-1)),
        'jerk'             : 1e+5 * th.mean(th.sum(th.square(th.diff(episode_data['xy'][:, :, 2:], 2, dim=1)), dim=-1)),
    })

def calculate_loss_kashefi(episode_data):
    # from Kashefi et al. (2025) bioRxiv https://doi.org/10.1101/2025.09.04.674069
    return _with_total({
        'position'         : 1e+0 * th.mean(th.abs(episode_data['xy'][:,:,:2] - episode_data['targets'][:,:,:2])),
        'speed'            : 1e-3 * th.mean(th.square(episode_data['xy'][:,:,2:])),
        'jerk'             : 1e-4 * th.mean(th.square(th.diff(episode_data['xy'][:,:,2:],n=2,dim=1))),
        'muscle'           : 1e-4 * th.mean(episode_data['force']),
        'muscle_derivative': 1e-4 * th.mean(th.square(th.diff(episode_data['force'],n=1,dim=1))),
        'hidden'           : 1e-2 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e-1 * th.mean(th.square(th.diff(episode_data['hidden'],n=2,dim=1))), # spectral
    })
