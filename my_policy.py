"""The networks: PolicyGRU (a plain GRU) and ModularPolicyGRU (a GRU divided into modules)."""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PolicyGRU(nn.Module):
    """A single-layer GRU followed by a sigmoid readout to the muscles."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.gru = th.nn.GRU(input_dim, hidden_dim, self.n_layers, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # the default initialization in torch isn't ideal
        th.nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        th.nn.init.orthogonal_(self.gru.weight_hh_l0)
        th.nn.init.zeros_(self.gru.bias_ih_l0)
        th.nn.init.zeros_(self.gru.bias_hh_l0)
        th.nn.init.xavier_uniform_(self.fc.weight)
        th.nn.init.constant_(self.fc.bias, -3.)

        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x[:, None, :], h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h

    def init_hidden(self, batch_size):
        return th.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device)


class ModularPolicyGRU(nn.Module):
    """A GRU whose hidden units are divided into modules (e.g. brain areas) with sparse connections.

    Each mask has one entry per module and gives the probability of a connection:
        vision_mask, proprio_mask, task_mask: from each kind of input to the module
        connectivity_mask[i, j]: the proportion of units in module j that project to module i
        output_mask: from the module to the outputs (muscles)
    Connections that are absent at initialization stay absent during training.
    freeze() stops chosen groups of weights from changing during any further training.

    Other arguments:
        module_size: number of units in each module
        vision_dim, proprio_dim, task_dim: which columns of the input hold each kind of input
        connectivity_delay[i, j]: delay (timesteps) from module j to module i
        output_delay: delay (timesteps) from hidden activity to output
        spectral_scaling: if given, rescale the recurrent weights to this spectral radius
        activation: 'tanh' or 'rect_tanh' (rectified tanh)
    """

    def __init__(self, input_size, module_size, output_size,
                 vision_mask, proprio_mask, task_mask, connectivity_mask, output_mask,
                 vision_dim, proprio_dim, task_dim, connectivity_delay,
                 spectral_scaling=None, input_gain=1., device=th.device("cpu"),
                 random_seed=None, activation='tanh', output_delay=0):
        super().__init__()

        hidden_size = sum(module_size)
        self.device = device
        self.num_modules = len(module_size)
        self.input_size = input_size
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.connectivity_delay = connectivity_delay
        self.output_delay = output_delay
        self.hidden_noise = 1e-3  # std of the noise added to each unit's input, independently per trial and timestep
        self.max_connectivity_delay = int(np.max(connectivity_delay))  # plain int keeps th.compile happy
        self.max_delay = int(np.max([self.max_connectivity_delay, self.output_delay]))
        self.h_buffer = []
        rng = np.random.default_rng(seed=random_seed or None)

        assert activation in ('tanh', 'rect_tanh')
        if activation == 'tanh':
            self.activation = th.tanh
        else:
            self.activation = lambda hidden: th.max(th.zeros_like(hidden), th.tanh(hidden))

        # Make sure that all sizes check out
        for mask in (vision_mask, proprio_mask, task_mask, output_mask):
            assert len(mask) == self.num_modules
        assert connectivity_mask.shape == (self.num_modules, self.num_modules)
        assert len(vision_dim) + len(proprio_dim) + len(task_dim) == input_size

        # GRU parameters. Each W acts on [input, hidden] concatenated.
        def gate_weights():
            return nn.Parameter(th.cat((
                nn.init.xavier_uniform_(th.empty(hidden_size, input_size), gain=input_gain),
                nn.init.normal_(th.empty(hidden_size, hidden_size), 0, 1 / np.sqrt(hidden_size)),
            ), dim=1))

        self.h0 = nn.Parameter(th.zeros(1, hidden_size))                           # initial hidden state
        self.Wz, self.bz = gate_weights(), nn.Parameter(th.zeros(hidden_size))     # update gate
        self.Wr, self.br = gate_weights(), nn.Parameter(th.zeros(hidden_size))     # reset gate
        self.Wh, self.bh = gate_weights(), nn.Parameter(th.zeros(hidden_size))     # candidate hidden state
        self.Y = nn.Parameter(nn.init.xavier_uniform_(th.empty(output_size, hidden_size), gain=1))  # output
        self.bY = nn.Parameter(nn.init.constant_(th.empty(output_size), -3.))

        # Indices of the hidden units belonging to each module
        edges = np.cumsum([0] + list(module_size))
        self.module_dims = [np.arange(edges[i], edges[i + 1]) for i in range(self.num_modules)]

        # Connection probabilities onto each hidden unit (rows), from [input, hidden] (columns)
        p_hidden = np.zeros((hidden_size, input_size + hidden_size), dtype=np.float32)
        for i, rows in enumerate(self.module_dims):
            p_hidden[np.ix_(rows, vision_dim)] = vision_mask[i]
            p_hidden[np.ix_(rows, proprio_dim)] = proprio_mask[i]
            p_hidden[np.ix_(rows, task_dim)] = task_mask[i]
            for j, cols in enumerate(self.module_dims):
                # A random subset of module j's units projects to every unit of module i
                p = connectivity_mask[i, j]
                if p > 0:
                    projecting = rng.choice(cols, size=int(np.ceil(p * len(cols))), replace=False)
                    p_hidden[np.ix_(rows, projecting + input_size)] = 1.0

        # Connection probabilities from each hidden unit to the outputs
        p_output = np.zeros((output_size, hidden_size), dtype=np.float32)
        for j, cols in enumerate(self.module_dims):
            p_output[:, cols] = output_mask[j]

        # Sample the binary masks (1 = connection exists). They are stored with the weights.
        # (The bias masks are all ones; they are kept so that saved weight files stay loadable.)
        def frozen(values):
            return nn.Parameter(th.tensor(values, dtype=th.float32), requires_grad=False)

        mask_hidden = rng.binomial(1, p_hidden)
        mask_output = rng.binomial(1, p_output)
        self.mask_Wz = frozen(mask_hidden)
        self.mask_Wr = frozen(mask_hidden)
        self.mask_Wh = frozen(mask_hidden)
        self.mask_Y = frozen(mask_output)
        self.mask_bz = frozen(np.ones(hidden_size))
        self.mask_br = frozen(np.ones(hidden_size))
        self.mask_bh = frozen(np.ones(hidden_size))
        self.mask_bY = frozen(np.ones(output_size))

        # Which weights can still learn (1 = trainable, 0 = fixed): everything, until freeze() is called.
        # (Plain tensors rather than parameters, so they are not part of the saved weights.)
        self.trainable = {name: th.ones_like(getattr(self, name))
                          for name in ('Wz', 'Wr', 'Wh', 'Y', 'bz', 'br', 'bh', 'bY')}

        # Remove the masked-out weights, and keep them at zero by masking their gradients.
        # The gradients of fixed weights are zeroed in the same way.
        for name in ('Wz', 'Wr', 'Wh', 'Y', 'bz', 'br', 'bh', 'bY'):
            param, mask = getattr(self, name), getattr(self, 'mask_' + name)
            with th.no_grad():
                param.mul_(mask)
            param.register_hook(lambda grad, mask=mask, name=name: grad * mask.data * self.trainable[name])

        # Optionally rescale the recurrent weights of Wh to a given spectral radius
        if spectral_scaling:
            with th.no_grad():
                Wh_input, Wh_hidden = th.split(self.Wh, [input_size, hidden_size], dim=1)
                eig_norm = th.max(th.abs(th.linalg.eigvals(Wh_hidden)))
                if eig_norm > 1e-6:
                    self.Wh.data = th.cat((Wh_input, spectral_scaling * (Wh_hidden / eig_norm)), dim=1)

        self.to(device)

    def freeze(self, connections=(), inputs=False, outputs=False, hidden_biases=False):
        """Fix groups of weights at their current values for all further training.

        connections: (sending module, receiving module) pairs of module indices, e.g. [(2, 1)]
                     fixes the weights from module 2 onto module 1
        inputs: fix the weights from the inputs (vision, proprioception, task) onto all hidden units
        outputs: fix the weights from the hidden units to the outputs, and the output biases
        hidden_biases: fix the biases of all hidden units (a unit has one bias per gate, shared by
                       its input and recurrent connections)
        Weights "onto" a unit are fixed in all three GRU gates (Wz, Wr, Wh). Freezing accumulates
        over calls, and only stops the gradients: the optimizer must not move the weights either
        (see ReachingModel.freeze, which also clears the optimizer's momentum).
        """
        for name in ('Wz', 'Wr', 'Wh'):
            if inputs:
                self.trainable[name][:, :self.input_size] = 0
            for sender, receiver in connections:
                rows = th.as_tensor(self.module_dims[receiver])
                cols = th.as_tensor(self.module_dims[sender]) + self.input_size
                self.trainable[name][rows[:, None], cols[None, :]] = 0
        if outputs:
            self.trainable['Y'].zero_()
            self.trainable['bY'].zero_()
        if hidden_biases:
            for name in ('bz', 'br', 'bh'):
                self.trainable[name].zero_()

    @th.compile(mode='max-autotune')
    def update_buffer(self, h_buffer, h_prev):
        # Push h_prev onto the front of the buffer of past hidden states, dropping the oldest
        return th.cat((h_prev.unsqueeze(-1), h_buffer[:, :, :-1]), dim=-1)

    @th.compile(mode='max-autotune')
    def forward(self, x, h_prev):
        self.h_buffer = self.update_buffer(self.h_buffer, h_prev)

        # If there are delays between modules we need to go module-by-module (this is slower)
        if self.max_connectivity_delay > 0:
            h_new = th.zeros_like(h_prev)
            for i in range(self.num_modules):
                # Prepare delayed hidden states for each module
                h_prev_delayed = th.zeros_like(h_prev)
                for j in range(self.num_modules):
                    h_prev_delayed[:, self.module_dims[j]] = self.h_buffer[:, self.module_dims[j],
                                                                           self.connectivity_delay[i, j]]
                concat = th.cat((x, h_prev_delayed), dim=1)
                z = th.sigmoid(F.linear(concat, self.Wz[self.module_dims[i], :], self.bz[self.module_dims[i]]))
                r = th.sigmoid(F.linear(concat, self.Wr, self.br))
                concat_hidden = th.cat((x, r * h_prev_delayed), dim=1)
                h_tilda = self.activation(F.linear(concat_hidden, self.Wh[self.module_dims[i], :],
                                                   self.bh[self.module_dims[i]]) +
                                          (th.randn(h_prev.shape[0], len(self.module_dims[i])) * self.hidden_noise))
                h_new[:, self.module_dims[i]] = (1 - z) * h_prev_delayed[:, self.module_dims[i]] + z * h_tilda

        # If there are no delays between modules we can do a single pass
        else:
            concat = th.cat((x, h_prev), dim=1)
            z = th.sigmoid(F.linear(concat, self.Wz, self.bz))                    # update gate
            r = th.sigmoid(F.linear(concat, self.Wr, self.br))                    # reset gate
            concat_hidden = th.cat((x, r * h_prev), dim=1)
            h_tilda = self.activation(F.linear(concat_hidden, self.Wh, self.bh) + (th.randn_like(h_prev) * self.hidden_noise))
            h_new = (1 - z) * h_prev + z * h_tilda

        # Output layer
        if self.output_delay == 0:
            y = th.sigmoid(F.linear(h_new, self.Y, self.bY))
        else:
            y = th.sigmoid(F.linear(self.h_buffer[:, :, self.output_delay-1], self.Y, self.bY))
        return y, h_new

    def init_hidden(self, batch_size):
        # Tile the learnable initial hidden state across the batch, and fill the buffer with it
        h0 = th.tile(self.activation(self.h0), (batch_size, 1))
        self.h_buffer = th.tile(h0.unsqueeze(dim=2), (1, 1, self.max_delay+1))
        return h0
