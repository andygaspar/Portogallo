import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FlNet(nn.Module):
    def __init__(self, input_size, num_flights, num_airlines, num_trades, couples_combs):
        super().__init__()
        self.couples_combs = couples_combs
        self.num_airlines = num_airlines
        self.num_flights = num_flights
        self.input_size = input_size
        self.num_trades = num_trades

        self.l1 = nn.Linear(self.input_size, self.couples_combs)

    def forward(self, state):
        return self.l1(state)

