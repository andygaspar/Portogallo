from torch import nn
import torch

from Training.Agents.attention.attentionDecoder import AttentionDecoder
from Training.Agents.attention.attentionEncoder import AttentionEncoder
sMax = nn.Softmax(dim=-1)

class AttentionCodec(nn.Module):
    def __init__(self, action_dim, hidden_dim, n_heads, n_attention_layers, context_dim):

        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._n_layers = n_attention_layers
        self._context_dim = context_dim

        self._encoder = AttentionEncoder(self._action_dim, self._hidden_dim, self._n_heads, self._n_layers)
        self._decoder = AttentionDecoder(self._context_dim, self._hidden_dim, self._n_heads)

    def encode(self, schedule):
        self._encoder(schedule)

    def get_action_probs(self, context, actions, mask):
        score = self._decoder(context, actions)
        non_zeros = torch.nonzero(mask - 1, as_tuple=True)
        if len(non_zeros) > 0:
            score[non_zeros, :] = 0
        return sMax(score)


