import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

sMax = nn.Softmax(dim=1)

class attentionNet(nn.Module):
    def __init__(self, output_dim, hidden_dim, schedule_entry_size, trade_size, n_entries, n_trades, l_rate, weight_decay=1e-4):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.schedule_entry_size = schedule_entry_size
        self.schedule_len = self.schedule_entry_size * n_entries
        self.trade_size = trade_size
        self.trades_len = self.schedule_len + self.trade_size * n_trades
        self.loss = 0
        self.bestLoss = 100_000_000

        self.numFlightTypes = 15
        self.numTrades = 6
        self.numAirlines = 4
        self.numCombs = 6
        self.numFlights = 16

        ETA_info_size = 1
        time_info_size = 1

        self.flightConvSize = self.numAirlines + self.numFlightTypes + time_info_size + ETA_info_size
        self.singleTradeSize = (self.numAirlines + self.numCombs) * 2

        torch.cuda.current_device()
        print("Running on GPU:", torch.cuda.is_available())

        self.schedule_embedding = nn.Sequential(nn.Linear(self.schedule_entry_size, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)).to(self.device)

        # self.firstConvTrades = nn.Linear(self.singleTradeSize, self.singleTradeSize * 2).to(self.device)
        # self.secondConvTrades = nn.Linear(self.singleTradeSize * 2, 8).to(self.device)
        #
        # self.firstConvCurrentTrade = nn.Linear(self.singleTradeSize, self.singleTradeSize * 2).to(self.device)
        # self.secondConvCurrentTrade = nn.Linear(self.singleTradeSize * 2, 8).to(self.device)

        self.trade_embedding = nn.Sequential(nn.Linear(self.trade_size, self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim)).to(self.device)

        #self.value_net = nn.Sequential(nn.Linear(3*self.hidden_dim, self.hidden_dim),
        self.value_net = nn.Sequential(nn.Linear((self.numTrades+1)*self.singleTradeSize, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, 4*self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(4*self.hidden_dim, 2*self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(2*self.hidden_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.output_dim)).to(self.device)

        self.value_net[-1].weight.data = torch.abs(self.value_net[-1].weight.data)
        self.value_net[-1].bias.data = torch.abs(self.value_net[-1].bias.data)

        params = list(self.schedule_embedding.parameters()) + list(self.trade_embedding.parameters()) + list(self.value_net.parameters())

        self.optimizer = optim.Adam(params, weight_decay=weight_decay, lr=l_rate)

    def forward(self, state):
        # state = state.reshape((-1, state.shape[-1]))
        # schedules = state[:, : self.schedule_len].to(self.device)
        # trades = state[:, self.schedule_len : self.trades_len].to(self.device)
        # current_trades = state[:, self.trades_len : ].to(self.device)
        # print(trades.to("cpu").numpy())
        # print(current_trades.to("cpu").numpy())
        flights, trades = torch.split(state, [self.flightConvSize * self.numFlights,
                                                             self.singleTradeSize * self.numTrades +
                                                             self.singleTradeSize], dim=-1)

        # schedules = schedules.reshape((schedules.shape[0], -1, self.schedule_entry_size)).to(self.device)
        # trades = trades.reshape((trades.shape[0], -1, self.trade_size)).to(self.device)
        #
        # # schedules_e = self.schedule_embedding(schedules).to(self.device)
        # trades_e = self.trade_embedding(trades).to(self.device)
        # current_trades_e = self.trade_embedding(current_trades).to(self.device)
        #
        # trades_w = torch.matmul(trades_e, current_trades_e.unsqueeze(2)).to(self.device)
        # trades_w = sMax(trades_w)
        # trades_attention = torch.matmul(trades_e.transpose(1, 2), trades_w).transpose(1,2).squeeze(1).to(self.device)

        # schedules_w = torch.matmul(schedules_e, trades_attention).to(self.device)
        # schedules_w = sMax(schedules_w)
        # schedules_attention = torch.matmul(schedules_e.transpose(1, 2), schedules_w).to(self.device)

        # value_in = torch.cat((schedules_attention, trades_attention), dim=1).transpose(1,2).squeeze(1).to(self.device)
        #value_in = torch.cat((value_in, current_trades_e), dim = 1).to(self.device)

        return self.value_net(trades)



    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state.to(self.device))
        # action = torch.zeros_like(scores)
        # action[torch.argmax(scores)] = 1

        return scores

    def update_weights(self, batch: tuple, gamma: float=1.0):
        criterion = torch.nn.MSELoss()

        states, next_states, masks, actions, rewards, dones = (element.to(self.device) for element in batch)


        self.zero_grad()
        curr_Q = self.forward(states)
        curr_Q = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()

        with torch.no_grad():
            next_Q = self.forward(next_states)
            next_Q[masks == 0] = -float('inf')
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

        loss = criterion(curr_Q, expected_Q)
        self.loss = self.loss*0.9 + 0.1*loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()


        if self.loss < self.bestLoss:
            torch.save(self.state_dict(), "air.pt")

        return loss

