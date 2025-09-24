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

        # define initial states, delay range, and catch trial percentage
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
            init_states = self.effector.draw_random_uniform_states(batch_size).detach().cpu().numpy() # random initial state

        # Vectorized delay time generation
        delay_tg_times = np.random.uniform(delay_tg[0] / self.dt, delay_tg[1] / self.dt, batch_size).astype(int)
        delay_go_times = np.random.uniform(delay_go[0] / self.dt, delay_go[1] / self.dt, batch_size).astype(int)
        
        # Vectorized catch trial determination
        is_catch = np.random.rand(batch_size) < catch_chance
        
        # Batch tensor conversions - compute all start points and final targets
        start_points = self.effector.joint2cartesian(th.tensor(init_states)).detach().cpu().numpy()
        
        if self.run_mode in ['test_center_out', 'train_center_out']:
            # Center-out targets: start_point + offset for each trial
            final_targets = start_points + offsets
        else:
            # Random targets for training mode - batch with start_points conversion
#            final_states = self.effector.draw_random_uniform_states(batch_size)
#            final_targets = self.effector.joint2cartesian(final_states).detach().cpu().numpy()
            n = np.shape(start_points)[0]
            final_targets = np.zeros((n,4))
            for i in range(n):
                found = False
                while not found:
                    tg_state = self.effector.draw_random_uniform_states(1)
                    tg_hand = self.effector.joint2cartesian(tg_state).detach().cpu().numpy()
                    hdist = (get_xy_dist(tg_hand[0][0:2], start_points[i,0:2]))
                    found = (hdist>=dmin) and (hdist<=dmax)
                final_targets[i,0:2] = tg_hand[0][0:2]

        # Create arrays for targets (for loss function) and inputs (for RNN)
        targets = np.zeros((batch_size, n_timesteps, start_points.shape[1]))
        inputs = np.zeros(shape=(batch_size, n_timesteps, 3))
        for i in range(batch_size):
            if not is_catch[i]:
                # inputs to RNN
                inputs[ i, :delay_tg_times[i], 0:2] = start_points[i, 0:2]
                inputs[ i, delay_tg_times[i]:, 0:2] = final_targets[i,0:2]
                inputs[ i, :delay_go_times[i],   2] = 0.0
                inputs[ i, delay_go_times[i]:,   2] = 1.0
                # targets for loss function calculation
                targets[i, :delay_go_times[i]+1,   :] = start_points[i]
                targets[i, delay_go_times[i]+1:,   :] = final_targets[i]
            elif is_catch[i]:
                # inputs to RNN
                inputs[ i, :delay_tg_times[i], 0:2] = start_points[i, 0:2]
                inputs[ i, delay_tg_times[i]:, 0:2] = final_targets[i,0:2]
                inputs[ i, :delay_go_times[i],   2] = 0.0
                inputs[ i, delay_go_times[i]:,   2] = 0.0
                # targets for loss function calculation
                targets[i, :delay_go_times[i]+1,   :] = start_points[i]
                targets[i, delay_go_times[i]+1:,   :] = start_points[i]
        
        # Add vectorized noise to all inputs at once
        noise = np.random.normal(loc=0., scale=1e-3, size=(batch_size, n_timesteps, 3))
        inputs += noise

        all_inputs = {"inputs": inputs}
        return [all_inputs, targets, init_states]
        #return [inputs, targets, init_states]


def generate_delay_time(delay_min, delay_max, delay_mode):#
    if delay_mode == 'random':
        delay_time = np.random.uniform(delay_min, delay_max)
    elif delay_mode == 'noDelayInput':
        delay_time = 0
    else:
        raise AttributeError

    return int(delay_time)


def get_xy_dist(hand1,hand2):
    # returns cartesian distance between two (xy) hand positions
    return np.sqrt(np.sum((hand2-hand1)**2))

