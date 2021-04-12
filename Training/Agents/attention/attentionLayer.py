import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, bias=True):
        super(AttentionLayer, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._bias = bias

        self._attentionLayer = nn.MultiheadAttention(self._hidden_dim, self._n_heads, bias=self._bias).to(self.device)
        self._bn1 = nn.BatchNorm1d(self._hidden_dim).to(self.device)
        self._ff = nn.Sequential(nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU()).to(self.device)
        self._bn2 = nn.BatchNorm1d(self._hidden_dim).to(self.device)

    def forward(self, x):
        x = x.unsqueeze(-2)
        y, _ = self._attentionLayer(x, x, x)
        h_hat = self._bn1((x + y).squeeze(-2))
        return self._bn2(h_hat + self._ff(h_hat))