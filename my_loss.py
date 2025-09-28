import numpy as np
import torch as th

loss_weights = {
    'position'          : 1e+3,
    'speed'             : 5e+1,
    'jerk'              : 1e+4,
    'muscle'            : 2e-1,
    'muscle_derivative' : 1e+0,
    'hidden'            : 2e+2,
    'hidden_derivative' : 1e+2
    }

def calculate_loss(episode_data):

    xy          = episode_data['xy'][:,:,0:2]
    speed       = episode_data['xy'][:,:,2:]
    tg          = episode_data['targets'][:,:,0:2]
    force       = episode_data['force']
    hidden      = episode_data['hidden']

    jerk        = th.diff(speed,                  n=2, dim=1)
    force_diff  = th.diff(episode_data['force'],  n=1, dim=1)
    hidden_diff = th.diff(episode_data['hidden'], n=2, dim=1)

    losses = {
        'position'         : loss_weights['position']          * th.mean(th.abs(xy - tg)),
        'speed'            : loss_weights['speed']             * th.mean(th.square(speed)),
        'jerk'             : loss_weights['jerk']              * th.mean(th.square(jerk)),
        'muscle'           : loss_weights['muscle']            * th.mean(force),
        'muscle_derivative': loss_weights['muscle_derivative'] * th.mean(th.square(force_diff)),
        'hidden'           : loss_weights['hidden']            * th.mean(th.square(hidden)),
        'hidden_derivative': loss_weights['hidden_derivative'] * th.mean(th.square(hidden_diff))
    }
    
    losses['total'] = sum(losses.values())
    return losses



