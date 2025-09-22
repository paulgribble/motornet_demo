import numpy as np
import torch as th

loss_weights = {
    'position'          : 1e+2,
    'speed'             : 1e-1,
    'jerk'              : 1e+2,
    'muscle'            : 1e-2,
    'muscle_derivative' : 1e-2,
    'hidden'            : 1e-0,
    'hidden_derivative' : 1e+1
    }


def calculate_loss(episode_data):

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


