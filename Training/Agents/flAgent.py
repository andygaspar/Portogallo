import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FlNet(nn.Module):
    def __init__(self, input_size, weight_decay, num_flight_types, num_airlines, num_flights, num_trades, num_combs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.numFlightTypes = num_flight_types
        self.numCombs = num_combs
        self.numAirlines = num_airlines
        self.numFlights = num_flights
        self.inputSize = input_size
        self.numTrades = num_trades
        self.loss = 0
        self.bestLoss = 100_000_000

        torch.cuda.current_device()
        print("Running on GPU:", torch.cuda.is_available())

        self.l1 = nn.Linear(self.inputSize, self.inputSize * 2).to(self.device)
        self.l2 = nn.Linear(self.inputSize * 2, self.inputSize * 2).to(self.device)
        self.l3 = nn.Linear(self.inputSize * 2, self.inputSize * 2).to(self.device)
        self.l4 = nn.Linear(self.inputSize * 2, self.inputSize * 2).to(self.device)
        self.l5 = nn.Linear(self.inputSize * 2, self.numCombs).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

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

    def update_weights(self, batch: tuple, gamma: float = 0.9):
        criterion = torch.nn.MSELoss()

        states, next_states, masks, actions, rewards, dones = (element.to(self.device) for element in batch)

        self.zero_grad()
        curr_Q = self.forward(states)
        curr_Q = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()

        with torch.no_grad():
            next_Q = self.forward(next_states)
            next_Q += masks
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

        loss = criterion(curr_Q, expected_Q)  # .detach()
        self.loss = loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()

        if self.loss < self.bestLoss:
            torch.save(self.state_dict(), "air.pt")


