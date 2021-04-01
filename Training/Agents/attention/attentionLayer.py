import torch
import torch.nn as nn
import torch.F as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, bias=True):
        super(AttentionLayer, self).__init__()

        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._bias = bias

        self._attentionLayer = nn.MultiheadAttention(self._hidden_dim, self._n_heads, bias=self._bias)
        self._bn1 = nn.BatchNorm1D(self._hidden_dim)
        self._ff = nn.Sequential(nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU())
        self._bn2 = nn.BatchNorm1D(self._hidden_dim)


    def forward(self, x):

        h_hat = self._bn1(x + self._attentionLayer(x))
        return self._bn2(h_hat + self._ff(h_hat))

