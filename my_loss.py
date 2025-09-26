import numpy as np
import torch as th

loss_weights = {
    'position'          : 1e+3,
    'speed'             : 1e+2, #1e-1
    'jerk'              : 1e+6, #1e+0
    'muscle'            : 1e-1,
    'muscle_derivative' : 1e+0,
    'hidden'            : 1e+1,
    'hidden_derivative' : 1e+3
    }


def calculate_loss_orig(episode_data):

    losses = {}

    losses['position']          = th.mean(th.sum(th.abs(episode_data['xy'][:,:,0:2]-episode_data['targets']), dim=-1))
    losses['speed']             = th.mean(th.sum(th.square(episode_data['xy'][:,:,2:]), dim=-1))
    losses['jerk']              = th.mean(th.sum(th.square(th.diff(episode_data['xy'][:,:,2:], n=2, dim=1)), dim=-1))
    losses['muscle']            = th.mean(th.sum(episode_data['force'], dim=-1))
    losses['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(episode_data['force'], n=1, dim=1)), dim=-1))
    losses['hidden']            = th.mean(th.sum(th.square(episode_data['hidden']), dim=-1))
    losses['hidden_derivative'] = th.mean(th.sum(th.square(th.diff(episode_data['hidden'], n=2, dim=1)), dim=-1)) # spectral note n=2

    # Weight the loss values in place
    for key in losses.keys():
        losses[key] *= loss_weights[key]

    # Calculate total directly
    losses['total'] = sum(losses[k] for k in losses.keys())

    return losses


def calculate_loss(episode_data):
    # Pre-compute common operations
    speed_data  = episode_data['xy'][:,:,2:]
    jerk_data   = th.diff(speed_data,             n=2, dim=1)
    force_diff  = th.diff(episode_data['force'],  n=1, dim=1)
    hidden_diff = th.diff(episode_data['hidden'], n=2, dim=1)
    
    # Calculate losses with gradients enabled
    losses = {
        'position'         : loss_weights['position']          * th.mean(th.abs(episode_data['xy'][:,:,0:2] - episode_data['targets'])),
        'speed'            : loss_weights['speed']             * th.mean(th.square(speed_data)),
        'jerk'             : loss_weights['jerk']              * th.mean(th.square(jerk_data)),
        'muscle'           : loss_weights['muscle']            * th.mean(episode_data['force']),
        'muscle_derivative': loss_weights['muscle_derivative'] * th.mean(th.square(force_diff)),
        'hidden'           : loss_weights['hidden']            * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': loss_weights['hidden_derivative'] * th.mean(th.square(hidden_diff))
    }
    
    losses['total'] = sum(losses.values())
    return losses



