import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class AirNet(nn.Module):
    def __init__(self, input_size, num_flights, num_airlines, num_trades):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_airlines = num_airlines
        self.num_flights = num_flights
        self.input_size = input_size
        self.num_trades = num_trades
        self.to(self.device)

        self.l1 = nn.Linear(self.input_size, self.num_airlines)

        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)

    def forward(self, state):
        return self.l1(state)

    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def update_weights(self, batch: tuple, gamma: float=0.9):
        criterion = torch.nn.MSELoss()

        states, next_states, actions, rewards, dones = batch

        for i in range(10):
            curr_Q = self.forward(states)
            curr_Q  = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()
            next_Q =self.forward(next_states)
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

            loss = criterion(curr_Q, expected_Q)  #.detach()
            self.loss = loss.item()
            #self.optimizer.zero_grad()
            self.zero_grad()
            loss.backward()
            print(loss)
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
