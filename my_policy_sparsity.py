import torch as th


class Policy(th.nn.Module):
    # includes sparsity matrix to structurally constrain connectivity

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device, sparsity: th.Tensor, freeze_output_layer=False, learn_h0=True):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.gru = th.nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # register sparsity mask as buffer (won't be a parameter)
        # sparsity is [H, H] with 0/1
        assert sparsity.shape == (hidden_dim, hidden_dim)
        mask_hh = sparsity.float().repeat(3, 1)  # [3H, H]
        self.register_buffer("hh_mask", mask_hh)

        if freeze_output_layer:
            for param in self.fc.parameters():
                param.requires_grad = False

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
                # apply sparsity mask to initial recurrent weights
                with th.no_grad():
                    param.mul_(self.hh_mask)
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
            
        # mask gradients for recurrent weights so zeros stay zero
        self.gru.weight_hh_l0.register_hook(
            lambda grad: grad * self.hh_mask
        )

        self.to(device)

    def apply_masks(self):
        # Call this after optimizer.step()
        with th.no_grad():
            self.gru.weight_hh_l0.mul_(self.hh_mask)

    def forward(self, x, h):
        y, h = self.gru(x[:, None, :], h)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h

    def init_hidden(self, batch_size):
        if hasattr(self, 'h0'):
            hidden = self.h0.repeat(1, batch_size, 1).to(self.device)
        else:
            weight = next(self.parameters()).data
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
