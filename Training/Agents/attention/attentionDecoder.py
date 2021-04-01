import torch
import torch.nn as nn
import numpy as np

class AttentionDecoder(nn.Module):
    def __init__(self, context_dim, action_dim, n_heads, C=10):

        super(AttentionDecoder, self).__init__()

        self._context_dim = context_dim
        self._action_dim = action_dim
        self._n_heads = n_heads
        self._C = C
        self._hidden_dim = int(self._action_dim / self._n_heads)

        self._queries = nn.ModuleList([nn.Linear(self._context_dim, self._hidden_dim, bias=False) for _ in range(self._n_heads)])
        self._keys = nn.ModuleList([nn.Linear(self._actions, self._hidden_dim, bias=False) for _ in range(self._n_heads)])
        self._values = nn.ModuleList([nn.Linear(self._actions, self._hidden_dim, bias=False) for _ in range(self._n_heads)])

        self._prob_queries = nn.Linear(self._action_dim, self._action_dim, bias=False)
        self._prob_keys = nn.Linear(self._action_dim, self._action_dim, bias=False)

    def forward(self, context, actions):
        queries = [q_i(context) for q_i in self._queries]
        keys = [k_i(actions) for k_i in self._keys]
        values = [v_i(actions) for v_i in self._values]

        u_vals = [torch.matmul(k, q.reshape((-1, self._action_dim, 1))) / np.sqrt(self._hidden_dim)
                  for k, q in zip(queries, keys)]

        c_embeddings = [torch.matmul(torch.transpose(u, -2, -1), v) for u, v in zip(u_vals, values)]
        c_embeddings = torch.cat(*c_embeddings, dim=-1)

        p_queries = self._prob_queries(c_embeddings)
        p_keys = self._prob_keys(actions)

        probs = torch.matmul(p_keys, torch.transpose(p_queries, -2, -1)) / np.sqrt(self._hidden_dim)
        probs = self._C * torch.tanh(probs)

        return probs
