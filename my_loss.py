import numpy as np
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


def calculate_loss_mehrdad(episode_data):
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

