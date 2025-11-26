import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.cm import jet
import numpy as np
import torch.nn as nn
import torch.nn.functional as F




class Policy(th.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.n_areas = 1
        
        self.gru = th.nn.GRU(input_dim, hidden_dim, self.n_layers, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # Learning and Loss parameters
        self.lr = 1e-3

        self.loss_act = 1e-5
        self.loss_hdn = 1e-6
        self.loss_hdn_diff = 1e-7
        self.loss_weight_decay = 1e-6
        self.loss_weight_sparsity = 1e-6
        self.loss_speed = 1e-6

        
        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                th.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError
        
        
        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x[:, None, :], h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    


class Policy_cnn(th.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device, **kwargs):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.n_areas = 1
        self.grid_size = kwargs.get('grid_size', 50)
        
        self.gru = th.nn.GRU(input_dim, hidden_dim, self.n_layers, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()
        # Convnet 
        self.conv1 = th.nn.Conv2d(in_channels=1, out_channels=5, kernel_size = 3, stride=1, padding=1)
        self.conv2 = th.nn.Conv2d(in_channels=5, out_channels=10, kernel_size = 3, stride=1, padding=1)
        self.conv3 = th.nn.Conv2d(in_channels=10, out_channels=15, kernel_size = 3, stride=1, padding=1)
        self.pool1 = th.nn.MaxPool2d(kernel_size=2)
        self.pool2 = th.nn.MaxPool2d(kernel_size=2)
        self.pool3 = th.nn.MaxPool2d(kernel_size=2)
        # Fully connected layers
        self.fcv = th.nn.Linear(15*6*6, 32)

        # Learning and Loss parameters
        self.lr = 1e-3

        self.loss_act = 1e-5
        self.loss_hdn = 1e-6
        self.loss_hdn_diff = 1e-7

        
        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                th.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            elif name == "conv1.bias":
                th.nn.init.constant_(param, -5.)
            elif name == "conv2.bias":
                th.nn.init.constant_(param, -5.)
            elif name == "conv3.bias":
                th.nn.init.constant_(param, -5.)
            elif name == "conv1.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "conv2.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "conv3.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fcv.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fcv.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError
        
        
        self.to(device)

    def forward(self, x, h0):
        x_visual_inst = x[:, None, :self.grid_size**2].reshape(shape=(-1, 1, self.grid_size, self.grid_size))
        x_feedback = x[:, None, self.grid_size**2:]

        x = th.nn.ReLU()(self.conv1(x_visual_inst))
        x = self.pool1(x)
        x = th.nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        x = th.nn.ReLU()(self.conv3(x))
        x = self.pool3(x)
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten
        x = th.nn.ReLU()(self.fcv(x))
        x = x[:, None,:]

        y, h = self.gru(th.cat((x, x_feedback), axis=-1), h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)

        return u, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    

    
class Policy_two_areas(th.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device, **kwargs):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.n_areas = 2
        
        self.gru0 = th.nn.GRU(input_dim  + int(hidden_dim/2), hidden_dim, self.n_layers, batch_first=True)
        self.gru1 = th.nn.GRU(hidden_dim, int(hidden_dim/2), self.n_layers, batch_first=True)
        self.fc = th.nn.Linear(int(hidden_dim/2), output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # Initial y1
        batch_size = kwargs.get('batch_size', 64)
        self.y1_loop = self.init_hidden(batch_size)[1].permute(1,0,2)

        # Learning and Loss parameters
        self.lr = 1e-3
        self.loss_act = 1e-5
        self.loss_hdn = 1e-6
        self.loss_hdn_diff = 1e-7
        self.loss_sparsity = 1e-6


    
        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if "weight_ih_l0" in name:
                th.nn.init.xavier_uniform_(param)
            elif "weight_hh_l0" in name:
                th.nn.init.orthogonal_(param)
            elif "bias_ih_l0" in name:
                th.nn.init.zeros_(param)
            elif "bias_hh_l0" in name:
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError
        
        
        self.to(device)

    def forward(self, x, h):
        y0, h0 = self.gru0(th.concat((x[:, None, :], self.y1_loop), axis=-1), h[0])
        y1, h1 = self.gru1(y0, h[1])
        self.y1_loop = y1.detach()
        u = self.sigmoid(self.fc(y1)).squeeze(dim=1)
        return u, [h0, h1]
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden0 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        hidden1 = weight.new(self.n_layers, batch_size, int(self.hidden_dim/2)).zero_().to(self.device)

        return [hidden0, hidden1]
    
class ModularPolicyGRU(nn.Module):
    def __init__(self, input_size: int, module_size: list, output_size: int,
                 vision_mask: list, proprio_mask: list, task_mask: list,
                 connectivity_mask: np.ndarray, output_mask: list,
                 vision_dim: list, proprio_dim: list, task_dim: list,
                 connectivity_delay: np.ndarray, spectral_scaling=None,
                 device=th.device("cpu"), random_seed=None, activation='tanh'):
        super(ModularPolicyGRU, self).__init__()

        # Store class info
        hidden_size = sum(module_size)
        assert activation == 'tanh' or activation == 'rect_tanh'
        if activation == 'tanh':
            self.activation = lambda hidden: th.tanh(hidden)
        elif activation == 'rect_tanh':
            self.activation = lambda hidden: th.max(th.zeros_like(hidden), th.tanh(hidden))
        self.spectral_scaling = spectral_scaling
        self.device = device
        self.num_modules = len(module_size)
        self.input_size = input_size
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.connectivity_delay = connectivity_delay
        self.max_delay = np.max(connectivity_delay)
        self.h_buffer = []

        # Set the random seed
        if random_seed:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = np.random.default_rng()

        # Make sure that all sizes check out
        assert len(vision_mask) == self.num_modules
        assert len(proprio_mask) == self.num_modules
        assert len(task_mask) == self.num_modules
        assert connectivity_mask.shape[0] == connectivity_mask.shape[1] == self.num_modules
        assert len(output_mask) == self.num_modules
        assert len(vision_dim) + len(proprio_dim) + len(task_dim) == self.input_size

        # Initialize all GRU parameters
        # Initial hidden state
        self.h0 = nn.Parameter(th.zeros(1, hidden_size))
        # Update gate
        self.Wz = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=1),
                                       nn.init.orthogonal_(th.Tensor(hidden_size, hidden_size))), dim=1))
        self.bz = nn.Parameter(th.zeros(hidden_size))
        # Reset gate
        self.Wr = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=1),
                                       nn.init.orthogonal_(th.Tensor(hidden_size, hidden_size))), dim=1))
        self.br = nn.Parameter(th.ones(hidden_size))
        # Candidate hidden state
        self.Wh = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=1),
                                       nn.init.orthogonal_(th.Tensor(hidden_size, hidden_size))), dim=1))
        self.bh = nn.Parameter(th.zeros(hidden_size))

        # Initialize all output parameters
        self.Y = nn.Parameter(nn.init.xavier_uniform_(th.Tensor(output_size, hidden_size), gain=1))
        self.bY = nn.Parameter(nn.init.constant_(th.Tensor(output_size), -5.))

        # Create indices for indexing modules
        module_dims = []
        for m in range(self.num_modules):
            if m > 0:
                module_dims.append(np.arange(module_size[m]) + module_dims[-1][-1] + 1)
            else:
                module_dims.append(np.arange(module_size[m]))
        self.module_dims = module_dims

        # Create sparsity mask for GRU
        h_probability_mask = np.zeros((hidden_size, input_size + hidden_size), dtype=np.float32)
        i_module = 0
        j_module = 0
        for i in range(self.Wz.shape[0]):
            for m in range(self.num_modules):
                if i in module_dims[m]:
                    i_module = m
            for j in range(self.Wz.shape[1]):
                module_type = 'hidden'
                if j < input_size:
                    if j in vision_dim:
                        module_type = 'vision'
                    elif j in proprio_dim:
                        module_type = 'proprio'
                    elif j in task_dim:
                        module_type = 'task'
                if module_type == 'hidden':
                    for m in range(self.num_modules):
                        if j in (module_dims[m] + input_size):
                            j_module = m
                        h_probability_mask[i, j] = connectivity_mask[i_module, j_module]
                elif module_type == 'vision':
                    h_probability_mask[i, j] = vision_mask[i_module]
                elif module_type == 'proprio':
                    h_probability_mask[i, j] = proprio_mask[i_module]
                elif module_type == 'task':
                    h_probability_mask[i, j] = task_mask[i_module]

        # Create sparsity mask for output
        y_probability_mask = np.zeros((output_size, hidden_size), dtype=np.float32)
        j_module = 0
        for j in range(self.Y.shape[1]):
            for m in range(self.num_modules):
                if j in module_dims[m]:
                    j_module = m
            y_probability_mask[:, j] = output_mask[j_module]

        # Initialize masks with desired sparsity
        mask_connectivity = self.rng.binomial(1, h_probability_mask)
        mask_output = self.rng.binomial(1, y_probability_mask)

        # Masks for weights and biases
        self.mask_Wz = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Wr = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Wh = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Y = nn.Parameter(th.tensor(mask_output), requires_grad=False)
        # No need to mask any biases for now
        self.mask_bz = nn.Parameter(th.ones_like(self.bz), requires_grad=False)
        self.mask_br = nn.Parameter(th.ones_like(self.br), requires_grad=False)
        self.mask_bh = nn.Parameter(th.ones_like(self.bh), requires_grad=False)
        self.mask_bY = nn.Parameter(th.ones_like(self.bY), requires_grad=False)

        # Zero out weights and biases that we don't want to exist
        self.Wz = nn.Parameter(th.mul(self.Wz, self.mask_Wz))
        self.Wr = nn.Parameter(th.mul(self.Wr, self.mask_Wr))
        self.Wh = nn.Parameter(th.mul(self.Wh, self.mask_Wh))
        self.Y = nn.Parameter(th.mul(self.Y, self.mask_Y))
        self.bz = nn.Parameter(th.mul(self.bz, self.mask_bz))
        self.br = nn.Parameter(th.mul(self.br, self.mask_br))
        self.bh = nn.Parameter(th.mul(self.bh, self.mask_bh))
        self.bY = nn.Parameter(th.mul(self.bY, self.mask_bY))

        Wh_i, Wh = th.split(self.Wh.detach(), [input_size, hidden_size], dim=1)
        _, mask_Wh = th.split(self.mask_Wh.detach(), [input_size, hidden_size], dim=1)
        Wh = self.orthogonalize_with_sparsity(Wh.numpy(), mask_Wh.numpy())
        self.Wh = nn.Parameter(th.mul(th.cat((Wh_i, th.tensor(Wh)), dim=1), self.mask_Wh))
        # Optional rescaling of Wh eigenvalues
        if self.spectral_scaling:
            Wh_i, Wh = th.split(self.Wh.detach(), [input_size, hidden_size], dim=1)
            eig_norm = th.max(th.real(th.linalg.eigvals(Wh)))
            Wh = self.spectral_scaling * (Wh / eig_norm)
            self.Wh = nn.Parameter(th.cat((Wh_i, Wh), dim=1))

        # Registering a backward hook to apply mask on gradients during backward pass
        self.Wz.register_hook(lambda grad: grad * self.mask_Wz.data)
        self.Wr.register_hook(lambda grad: grad * self.mask_Wr.data)
        self.Wh.register_hook(lambda grad: grad * self.mask_Wh.data)
        self.bz.register_hook(lambda grad: grad * self.mask_bz.data)
        self.br.register_hook(lambda grad: grad * self.mask_br.data)
        self.bh.register_hook(lambda grad: grad * self.mask_bh.data)
        self.Wh.register_hook(lambda grad: grad * self.mask_Wh.data)
        self.Y.register_hook(lambda grad: grad * self.mask_Y.data)
        self.bY.register_hook(lambda grad: grad * self.mask_bY.data)

        self.to(device)

    def forward(self, x, h_prev):
        # If there are delays between modules we need to go module-by-module (this is slow)
        if self.max_delay > 0:
            # Update hidden state buffer
            self.h_buffer[:, :, 1:] = self.h_buffer[:, :, 0:-1]
            self.h_buffer[:, :, 0] = h_prev
            # Forward pass
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
                                                   self.bh[self.module_dims[i]]))
                h = (1 - z) * h_prev_delayed[:, self.module_dims[i]] + z * h_tilda
                # Store new hidden states to correct module
                h_new[:, self.module_dims[i]] = h

        # If there are no delays between modules we can do a single pass
        else:
            concat = th.cat((x, h_prev), dim=1)
            z = th.sigmoid(F.linear(concat, self.Wz, self.bz))
            r = th.sigmoid(F.linear(concat, self.Wr, self.br))
            concat_hidden = th.cat((x, r * h_prev), dim=1)
            h_tilda = self.activation(F.linear(concat_hidden, self.Wh, self.bh))
            h_new = (1 - z) * h_prev + z * h_tilda

        # Output layer
        y = th.sigmoid(F.linear(h_new, self.Y, self.bY))
        return y, h_new

    def init_hidden(self, batch_size):
        # Tile learnable hidden state
        h0 = th.tile(self.activation(self.h0), (batch_size, 1))
        # Create initial hidden state buffer if needed
        if self.max_delay > 0:
            self.h_buffer = th.tile(h0.unsqueeze(dim=2), (1, 1, self.max_delay+1))
        return h0

    def orthogonalize_with_sparsity(self, matrix, sparsity_matrix):
        # Ensure sparsity_matrix is binary (0 or 1)
        assert np.all(np.isin(sparsity_matrix, [0, 1]))

        # Copy matrix so as not to modify the original
        Q = matrix.copy()

        # Loop over columns
        for i in range(Q.shape[1]):
            # Subtract projections onto previous columns
            for j in range(i):
                # Check if the column is a zero vector
                if np.dot(Q[:, j], Q[:, j]) < 1e-10:
                    continue

                proj = np.dot(Q[:, j], Q[:, i]) / np.dot(Q[:, j], Q[:, j])
                Q[:, i] -= proj * Q[:, j]

                # Reset the undesired entries to zero using sparsity matrix
                Q[:, i] *= sparsity_matrix[:, i]

            # Normalize current column, avoiding division by zero
            norm = np.linalg.norm(Q[:, i])
            if norm > 1e-10:
                Q[:, i] /= norm

                # Reset the undesired entries to zero again to ensure structure
                Q[:, i] *= sparsity_matrix[:, i]

        return Q

class VisionCNN(nn.Module):
    def __init__(self):
        super(VisionCNN, self).__init__()

        # Encoder
        self.enc_fc1 = nn.Linear(6, 49)
        self.enc_conv1 = nn.ConvTranspose3d(1, 32, kernel_size=(1,5,5), stride=(1,2,2))
        nn.init.xavier_uniform_(self.enc_conv1.weight)
        self.enc_conv2 = nn.ConvTranspose3d(32, 128, kernel_size=(1, 10, 10), stride=1)
        nn.init.xavier_uniform_(self.enc_conv2.weight)
        self.enc_conv3 = nn.ConvTranspose3d(128, 1, kernel_size=(1, 25, 25), stride=1)
        nn.init.xavier_uniform_(self.enc_conv3.weight)

        # Decoder
        self.dec_conv1 = nn.Conv3d(1, 128, kernel_size=(1, 3, 3), stride=1, padding='same')
        nn.init.xavier_uniform_(self.dec_conv1.weight)
        self.dec_conv2 = nn.Conv3d(128, 32, kernel_size=(1, 3, 3), stride=1, padding='same')
        nn.init.xavier_uniform_(self.dec_conv2.weight)
        self.dec_conv3 = nn.Conv3d(32, 1, kernel_size=(1, 3, 3), stride=1, padding='same')
        nn.init.xavier_uniform_(self.dec_conv3.weight)
        self.dec_fc1 = nn.Linear(144, 32)
        self.dec_fc2 = nn.Linear(32, 6)

    def forward(self, x):
        #print(x.shape)
        time_steps = x.shape[1]
        x = nn.Flatten(2,3)(x)   # Flatten the input [B, 6]
        #print(x.shape)
        # Encoding
        #print(x.shape)
        x = self.enc_fc1(x)
        #print(x.shape)
        x = th.reshape(x, (-1, time_steps, 7, 7))
        #print(x.shape)
        x = th.unsqueeze(x, axis=1)
        #print(x.shape)
        x = F.relu(self.enc_conv1(x))
        #print(x.shape)
        x = F.relu(self.enc_conv2(x))
        #print(x.shape)
        x = F.relu(self.enc_conv3(x))
        x_grid = x
        #print(x.shape)
        #print('===============')
        #print(x.shape)
        x = F.relu(self.dec_conv1(x))
        x = nn.MaxPool3d(kernel_size=(1, 2, 2))(x)
        #print(x.shape)
        x = F.relu(self.dec_conv2(x))
        x = nn.MaxPool3d(kernel_size=(1, 2, 2))(x)
        #print(x.shape)
        x = F.relu(self.dec_conv3(x))
        #rint(x.shape)
        x = nn.Flatten(3,4)(x)
        x = th.squeeze(x, axis=1)
        #print(x.shape)
        x = F.relu(self.dec_fc1(x))
        x_dec = x
        x = self.dec_fc2(x)
        #print(x.shape)
        x = th.reshape(x, (-1, time_steps, 3, 2))
        #print(x.shape)

        return x, x_dec, x_grid
