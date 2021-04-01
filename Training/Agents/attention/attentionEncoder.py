import torch
import torch.nn as nn
import attentionLayer.AttentionLayer


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, bias=False):
        super(AttentionEncoder, self).__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._bias = bias

        self._initEmbedding = nn.Linear(self._input_dim, self._hidden_dim)
        self._attentionLayers = nn.Sequential(*[AttentionLayer(self._hidden_dim, self._n_heads, bias = self._bias) for _ in range(self._n_layers)])


    def forward(self, x):
        return self._attentionLayers(self._initEmbedding(x))
