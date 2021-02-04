import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FlNet(nn.Module):
    def __init__(self, input_size, num_flights, num_airlines, num_trades, couples_combs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.couples_combs = couples_combs
        self.num_airlines = num_airlines
        self.num_flights = num_flights
        self.input_size = input_size
        self.num_trades = num_trades
        self.to(self.device)
        torch.cuda.current_device()
        print("Running on GPU:", torch.cuda.is_available())

        self.l1 = nn.Linear(self.input_size, self.couples_combs)

    def forward(self, state):
        return self.l1(state)

    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def update_weights(self, batch: tuple, gamma: float = 0.9):
        criterion = torch.nn.MSELoss()

        states, next_states, actions, rewards, dones = batch

        for i in range(10):
            curr_Q = self.forward(states)
            curr_Q = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()
            next_Q = self.forward(next_states)
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

            loss = criterion(curr_Q, expected_Q)  # .detach()
            self.loss = loss.item()
            # self.optimizer.zero_grad()
            self.zero_grad()
            loss.backward()
            print(loss)
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()

