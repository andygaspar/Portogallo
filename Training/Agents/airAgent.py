import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class AirNet(nn.Module):
    def __init__(self, input_size, weight_decay ,num_flight_types, num_airlines, num_flights, num_trades, num_combs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.numFlightTypes = num_flight_types
        self.numCombs = num_combs
        self.numAirlines = num_airlines
        self.numFlights = num_flights
        self.inputSize = input_size
        self.num_trades = num_trades
        self.loss = 0
        self.bestLoss = 100_000_000

        self.l1 = nn.Linear(self.inputSize, self.inputSize * 2).to(self.device)
        self.l2 = nn.Linear(self.inputSize * 2, self.inputSize * 2).to(self.device)
        self.l3 = nn.Linear(self.inputSize * 2, self.inputSize * 2).to(self.device)
        self.l4 = nn.Linear(self.inputSize * 2, self.inputSize * 2).to(self.device)
        self.l5 = nn.Linear(self.inputSize * 2, self.numAirlines).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state.to(self.device))
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1

        return action

    def update_weights(self, batch: tuple, gamma: float=0.9):
        criterion = torch.nn.MSELoss()

        states, next_states, masks, actions, rewards, dones = (element.to(self.device) for element in batch)


        self.zero_grad()
        curr_Q = self.forward(states)
        curr_Q = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()

        with torch.no_grad():
            next_Q = self.forward(next_states)
            next_Q[masks == 0] = -100
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

        loss = criterion(curr_Q, expected_Q)  #.detach()
        self.loss = loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()


        # if self.loss < self.bestLoss:
        #     torch.save(self.state_dict(), "air.pt")

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
