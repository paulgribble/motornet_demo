import numpy as np
import torch as th

# Here we are defining the Task. This is what we update to change the tasks the network has to learn.

class ExperimentTask:
    def __init__(self, effector, **kwargs):
        self.effector = effector
        self.dt = self.effector.dt
        self.delay_tg = kwargs.get('delay_tg', [0.3, 0.8]) # target cue delay period
        self.delay_go = kwargs.get('delay_go', [0.8, 1.3]) # go cue delay period
        self.dmin = kwargs.get("dmin", 0)      # min hand distance between targets (m)
        self.dmax = kwargs.get("dmax", np.inf) # max hand distance between targets (m)
        self.run_mode = kwargs.get('run_mode', 'train')    # {train, test_center_out, train_center_out}

    def generate(self, batch_size, n_timesteps, dmin=0, dmax=np.inf, **kwargs):
        # generate inputs to RNN, targets for use in loss function, and initial states
        # dmin and dmax are bounds on min and max target distance for the 'train' random target selection

        base_joint = np.deg2rad([50., 90., 0., 0.]).astype(np.float32) # shoulder, elbow angles

        # center-out targets
        radius = 0.10 # (m)
        n = batch_size
        angle = np.linspace(0,2*np.pi,n, endpoint=False)
        offsets = radius*np.array([np.cos(angle), np.sin(angle), np.zeros(n), np.zeros(n)]).T

        # define initial states (joint angles and vels), delay range, and catch trial percentage
        if self.run_mode == 'test_center_out': # test on center-out movements
            catch_chance = 0.     # no catch trials
            delay_tg = [0.5, 0.5] # target cue
            delay_go = [1.0, 1.0] # go cue
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
        elif self.run_mode == 'train_center_out': # train center-out movements
            catch_chance = 0.5    # 50% no-go catch trials
            delay_tg = self.delay_tg
            delay_go = self.delay_go
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
        else: # must be 'train' (train on random reaches across entire workspace)
            catch_chance = 0.5    # 50% no-go catch trials
            delay_tg = self.delay_tg
            delay_go = self.delay_go
#            # define initial joint angles
#            init_states = self.effector.draw_random_uniform_states(batch_size).detach().cpu().numpy() # random initial state
            # define initial joint angles within some bounds
            joints_min = np.array([20, 20])  * np.pi / 180 # shoulder, elbow (rad)
            joints_max= np.array([110, 110]) * np.pi / 180 # shoulder, elbow (rad)
            rnd = np.random.rand(batch_size,2)
            pos = (joints_max-joints_min) * rnd + joints_min
            vel = np.zeros((batch_size,2))
            init_states = np.hstack([pos, vel])

        # Vectorized delay time generation
        delay_tg_times = np.random.uniform(delay_tg[0] / self.dt, delay_tg[1] / self.dt, batch_size).astype(int)
        delay_go_times = np.random.uniform(delay_go[0] / self.dt, delay_go[1] / self.dt, batch_size).astype(int)
        
        # Vectorized catch trial determination
        is_catch = np.random.rand(batch_size) < catch_chance
        
        # Compute start points and movement targets (hand positions)
        start_points = self.effector.joint2cartesian(th.tensor(init_states)).detach().cpu().numpy()
        
        if self.run_mode in ['test_center_out', 'train_center_out']:
            # Center-out targets: start_point + offset for each trial
            final_targets = start_points + offsets
        else:
#            # Random targets for training mode - batch with start_points conversion
#            final_states  = self.effector.draw_random_uniform_states(batch_size)
#            final_targets = self.effector.joint2cartesian(final_states).detach().cpu().numpy()

            # set random end targets that are within [dmin, dmax] hand distance from start_points
            n = np.shape(start_points)[0]
            final_targets = np.zeros((n,4))
            for i in range(n):
                found = False
                while not found:
                    tg_state = self.effector.draw_random_uniform_states(1).detach().cpu().numpy()[0] # joint angles, vels
                    tg_hand = self.effector.joint2cartesian(tg_state).detach().cpu().numpy()[0]  # hand xy
                    hdist = np.sqrt(np.sum(np.square(tg_hand[0:2] - start_points[i,0:2]))) # dist from start
                    found = (hdist>=dmin) and (hdist<=dmax) # within desired distance range
                    found = found and all(tg_state[0:2] > joints_min) and all(tg_state[0:2] < joints_max) # within joint range
                    found = found and (tg_hand[1] > 0)   # hand not behind shoulder
                final_targets[i,0:2] = tg_hand[0:2]

        # Create arrays for targets (for loss function) and inputs (for RNN)
        targets = np.zeros((batch_size, n_timesteps, start_points.shape[1])) # initialize array to zeros
        inputs = np.zeros(shape=(batch_size, n_timesteps, 3))                # initialize array to zeros
        for i in range(batch_size):
            # inputs to RNN
            inputs[ i, :delay_tg_times[i], 0:2] = start_points[i, 0:2] # RNN always sees start tgt
            inputs[ i, delay_tg_times[i]:, 0:2] = final_targets[i,0:2] # and then final target
            inputs[ i, :delay_go_times[i],   2] = 0.0                  # and don't go until delay_go
            # targets for loss function calculation
            targets[i, :delay_go_times[i],   :] = start_points[i]
            if not is_catch[i]:
                inputs[ i, delay_go_times[i]:,   2] = 1.0               # go!
                targets[i, delay_go_times[i]:,   :] = final_targets[i]  # for loss function: we want to go to tgt
            elif is_catch[i]:
                inputs[ i, delay_go_times[i]:,   2] = 0.0               # don't go!
                targets[i, delay_go_times[i]:,   :] = start_points[i]   # for loss function: we want to stay at start
        
        # Add vectorized noise to all inputs at once
        noise = np.random.normal(loc=0., scale=1e-3, size=(batch_size, n_timesteps, 3))
        inputs += noise

        all_inputs = {"inputs": inputs}
        return [all_inputs, targets, init_states, delay_go_times]


def generate_delay_time(delay_min, delay_max, delay_mode):#
    if delay_mode == 'random':
        delay_time = np.random.uniform(delay_min, delay_max)
    elif delay_mode == 'noDelayInput':
        delay_time = 0
    else:
        raise AttributeError

    return int(delay_time)

