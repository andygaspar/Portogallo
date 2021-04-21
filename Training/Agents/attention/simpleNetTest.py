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
                                 nn.Linear(self._hidden_dim, self._hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self._hidden_dim, num_flights)).to(self.device)

        #self.optimizer = optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, state, masker):
        score = self._ff(state)
        probs = torch.zeros_like(score)
        non_zeros = torch.nonzero(masker.mask)
        if len(non_zeros) > 0:
            score = score[non_zeros]
        probs_valid = sMax(score)
        probs[non_zeros] = probs_valid
        return probs

    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state.to(self.device))
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1

        return action

    def update_weights(self, batch: tuple, gamma: float=0.9):
        criterion = torch.nn.MSELoss()

        states, next_states, actions, rewards, dones = (element.to(self.device) for element in batch)

        for i in range(10):
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
    #
    #   super().__init__()
    #   self.convSchedule = nn.Linear(21, 4)
    #   self.
    #   self.convSchedule = nn.Linear(5, 1)
    #
    #   self.tradesConv1 = nn.Linear()
    #
    #   self.fc1 = nn.Linear(20, 64)
    #   self.fc2 = nn.Linear(64, 4)
    #
    # def forward(self, air, others, trades):
    #
    #   first_conv_out = torch.split(air, [17 for _ in range(5)], dim=-1)
    #   second_conv_out = [F.relu(self.convAir1(t)) for t in first_conv_out]
    #   to_merge = torch.cat([F.relu(self.convAir2(t)) for t in second_conv_out], dim=-1)
    #
    #   first_conv_out = torch.split(others, [21 for _ in range(15)], dim=-1)
    #   second_conv_out = [F.relu(self.convOthers1(t)) for t in first_conv_out]
    #   to_merge1 = torch.cat([F.relu(self.convOthers2(t)) for t in second_conv_out], dim=-1)
    #
    #   merged_tensor = torch.cat((to_merge, to_merge1), dim=-1)
    #
    #   out = F.relu(self.fc1(merged_tensor))
    #
    #   # return self.fc2(out)
