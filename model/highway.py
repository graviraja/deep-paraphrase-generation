'''This code contains the implementation of highway network.

'''
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super().__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        """
        Args:
            x : input with shape of [batch_size, size]
        Returns:
            tensor with shape of [batch_size, size] after applying highway network
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """

        for layer in range(self.num_layers):
            # σ(x)
            gate = F.sigmoid(self.gate[layer](x))

            # f(G(x))
            non_linear = self.f(self.nonlinear[layer](x))

            # Q(x)
            linear = self.linear[layer](x)

            # σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x))
            x = gate * non_linear + (1 - gate) * linear

        return x
