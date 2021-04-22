import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
sMax = torch.nn.Softmax(dim=0)


class AgentNetwork(nn.Module):

    def __init__(self, weight_decay, num_flights):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.to(self.device)

        self.numFlights = num_flights
        self._hidden_dim = 64
        self.loss = 0

        self._ff = nn.Sequential(nn.Linear(num_flights * 2 + 2 * 5, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, self._hidden_dim*2),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim*2, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, num_flights)).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, state):
        score = self._ff(state)
        return score

    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state.to(self.device))
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1

        return action

    def update_weights(self, batch: tuple, gamma: float = 1):
        criterion = torch.nn.MSELoss()

        states, next_states, masks, actions, rewards, dones= (element.to(self.device) for element in batch)


        self.zero_grad()
        curr_Q = self.forward(states)
        curr_Q  = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()
        next_Q =self.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

        loss = criterion(curr_Q, expected_Q)  #.detach()
        self.loss = loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

   #solution: [[array([FA8, FA17], dtype=object), array([FB13, FB10], dtype=object), array([FC4, FC11], dtype=object)]]

#FD0,       FA1,      FD2,        FB6,     FC7,     FD5,     FB12,   FC4,
# #[40.5156, 40.2933, 40.4314,    -inf, 45.9346, 44.2575, 41.6733, 44.0008,
#          FA9,       FA8,      FB13,   FC15,   FB3,    FB10,     FD14,    FC19,
#         41.6020, 41.2491, 41.9072, 44.4279, 42.6854, 44.4565, 43.7079, 44.7275,
#          FA17,    FA16,    FD18,    FC11]
#         41.2276, 45.0535, 45.5708, 46.2582]