import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

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

        torch.cuda.current_device()
        print("Running on GPU:", torch.cuda.is_available())

        self.schedule_embedding = nn.Sequential(nn.Linear(self.schedule_entry_size, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)).to(self.device)

        self.trade_embedding = nn.Sequential(nn.Linear(self.trade_size, self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim)).to(self.device)

        self.value_net = nn.Sequential(nn.Linear(3*self.hidden_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.output_dim)).to(self.device)

        params = list(self.schedule_embedding.parameters()) + list(self.trade_embedding.parameters()) + list(self.value_net.parameters())

        self.optimizer = optim.Adam(params, weight_decay=weight_decay, lr=l_rate)

    def forward(self, state):
        state = state.reshape((-1, state.shape[-1]))
        schedules = state[:, : self.schedule_len].to(self.device)
        trades = state[:, self.schedule_len : self.trades_len].to(self.device)
        current_trades = state[:, self.trades_len : ].to(self.device)

        schedules = schedules.reshape((schedules.shape[0], -1, self.schedule_entry_size)).to(self.device)
        trades = trades.reshape((trades.shape[0], -1, self.trade_size)).to(self.device)

        schedules_e = self.schedule_embedding(schedules).to(self.device)
        trades_e = self.trade_embedding(trades).to(self.device)
        current_trades_e = self.trade_embedding(current_trades).to(self.device)

        trades_w = torch.matmul(trades_e, current_trades_e.unsqueeze(2)).to(self.device)
        trades_w = sMax(trades_w)
        trades_attention = torch.matmul(trades_e.transpose(1, 2), trades_w).to(self.device)

        schedules_w = torch.matmul(schedules_e, trades_attention).to(self.device)
        schedules_w = sMax(schedules_w)
        schedules_attention = torch.matmul(schedules_e.transpose(1, 2), schedules_w).to(self.device)

        value_in = torch.cat((schedules_attention, trades_attention), dim=1).transpose(1,2).squeeze(1).to(self.device)
        value_in = torch.cat((value_in, current_trades_e), dim = 1).to(self.device)

        return self.value_net(value_in).squeeze()



    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state.to(self.device))
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1

        return action

    def update_weights(self, batch: tuple, gamma: float=1.0):
        criterion = torch.nn.MSELoss()

        states, next_states, masks, actions, rewards, dones = (element.to(self.device) for element in batch)


        self.zero_grad()
        curr_Q = self.forward(states)
        curr_Q = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()

        with torch.no_grad():
            next_Q = self.forward(next_states)
            next_Q[masks == 0] = -1
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

        loss = criterion(curr_Q, expected_Q)  #.detach()
        self.loss = loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()


        if self.loss < self.bestLoss:
            torch.save(self.state_dict(), "air.pt")

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
