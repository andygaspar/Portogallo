import torch
import torch.nn as nn
from Training.Agents.attention.attentionLayer import AttentionLayer


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, bias=False):
        super(AttentionEncoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._bias = bias

        self._initEmbedding = nn.Linear(self._input_dim, self._hidden_dim).to(self.device)
        self._attentionLayers = nn.Sequential(
            *[AttentionLayer(self._hidden_dim, self._n_heads, bias=self._bias) for _ in range(self._n_layers)]).to(self.device)

    def forward(self, x):
        embeddings = self._initEmbedding(x)
        return self._attentionLayers(embeddings)