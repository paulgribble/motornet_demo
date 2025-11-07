import numpy as np
import torch as th

def calculate_loss(episode_data):

    losses = {
        'position'         : 1e+3 * th.mean(th.abs(episode_data['xy'][:,:,:2] - episode_data['targets'][:,:,:2])),
        'speed'            : 5e+1 * th.mean(th.square(episode_data['xy'][:,:,2:])),
        'jerk'             : 1e+4 * th.mean(th.square(th.diff(episode_data['xy'][:,:,2:],n=2,dim=1))),
        'muscle'           : 2e-1 * th.mean(episode_data['force']),
        'muscle_derivative': 1e+0 * th.mean(th.square(th.diff(episode_data['force'],n=1,dim=1))),
        'hidden'           : 2e+2 * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': 1e+2 * th.mean(th.square(th.diff(episode_data['hidden'],n=2,dim=1))) # spectral
    }
    
    losses['total'] = losses['position'] + losses['speed'] + losses['jerk'] + \
                      losses['muscle']   + losses['muscle_derivative'] + \
                      losses['hidden']   + losses['hidden_derivative']
    return losses

