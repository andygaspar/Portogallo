from torch import nn
import torch

from Training.Agents.attention.attentionDecoder import AttentionDecoder
from Training.Agents.attention.attentionEncoder import AttentionEncoder
sMax = nn.Softmax(dim=0)


class AttentionCodec(nn.Module):
    def __init__(self, action_dim, hidden_dim, n_heads, n_attention_layers, context_dim):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._n_layers = n_attention_layers
        self._context_dim = context_dim

        self._encoder = AttentionEncoder(self._action_dim, self._hidden_dim, self._n_heads, self._n_layers).to(self.device)
        self._decoder = AttentionDecoder(self._context_dim, self._hidden_dim, self._n_heads).to(self.device)

    def encode(self, schedule):
        return self._encoder(schedule.to(self.device))

    def get_action_probs(self, context, actions, mask):
        context = context.to(self.device)
        actions = actions.to(self.device)
        mask = mask.to(self.device)

        score = self._decoder(context, actions)
        non_zeros = torch.nonzero(mask - 1)
        if len(non_zeros) > 0:
            score[non_zeros, :] = -float('inf')
        probs = sMax(score)
        return probs