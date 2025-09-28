import numpy as np
import torch as th

loss_weights = {
    'position'          : 1e+3,
    'speed'             : 0e+0, # 5e+1
    'jerk'              : 1e+4,
    'muscle'            : 1e-1,
    'muscle_derivative' : 1e+0,
    'hidden'            : 1e+2,
    'hidden_derivative' : 1e+2
    }

def calculate_loss(episode_data):
    # Pre-compute common operations
    speed_data  = episode_data['xy'][:,:,2:]
    jerk_data   = th.diff(speed_data,             n=2, dim=1)
    force_diff  = th.diff(episode_data['force'],  n=1, dim=1)
    hidden_diff = th.diff(episode_data['hidden'], n=2, dim=1)

    # Calculate losses with gradients enabled

    xy = episode_data['xy'][:,:,0:2]
    tg = episode_data['targets'][:,:,0:2]
    pos_err = xy - tg

    # set to zero pos_err during desired movement interval
    des_mov_time = 0.500 # seconds
    des_mov_step = int(des_mov_time / episode_data['dt'])
    go_steps = episode_data['delay_go_times']
    go_mask = go_steps[:,None] + np.arange(des_mov_step)
    pos_err[:,go_mask,:] = 0.0

    losses = {
        'position'         : loss_weights['position']          * th.mean(th.abs(pos_err)),
        'speed'            : loss_weights['speed']             * th.mean(th.square(speed_data)),
        'jerk'             : loss_weights['jerk']              * th.mean(th.square(jerk_data)),
        'muscle'           : loss_weights['muscle']            * th.mean(episode_data['force']),
        'muscle_derivative': loss_weights['muscle_derivative'] * th.mean(th.square(force_diff)),
        'hidden'           : loss_weights['hidden']            * th.mean(th.square(episode_data['hidden'])),
        'hidden_derivative': loss_weights['hidden_derivative'] * th.mean(th.square(hidden_diff))
    }
    
    losses['total'] = sum(losses.values())
    return losses



