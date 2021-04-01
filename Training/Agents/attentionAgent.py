import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

sMax = nn.Softmax(dim=-1)


class AttentionNet(nn.Module):
    def __init__(self, hidden_dim, len_discretisation, l_rate,
                 weight_decay=1e-4):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.hidden_dim = hidden_dim
        self.flightDiscretisation = len_discretisation
        self.fl_repr_size = 36
        self.loss = 0
        self.bestLoss = 100_000_000

        torch.cuda.current_device()
        print("Running on GPU:", torch.cuda.is_available())

        self.schedule_embedding = nn.Sequential(nn.Linear(self.flightDiscretisation+1, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.fl_repr_size, bias=False)).to(self.device)

        self.attention_ff = nn.Sequential(nn.Linear(self.fl_repr_size, self.fl_repr_size),
                                          nn.ReLU(),
                                          nn.Linear(self.fl_repr_size, self.fl_repr_size)).to(self.device)

        self.action_attention_ff = nn.Sequential(nn.Linear(self.fl_repr_size, self.fl_repr_size),
                                                 nn.ReLU(),
                                                 nn.Linear(self.fl_repr_size, self.fl_repr_size)).to(self.device)

        self.action_embedding = nn.Sequential(nn.Linear(self.flightDiscretisation, self.hidden_dim, bias=False),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.fl_repr_size, bias=False)).to(self.device)

        self.value_net = nn.Sequential(nn.Linear(self.fl_repr_size, self.hidden_dim * 2),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, 16),
                                       nn.ReLU(),
                                       nn.Linear(16, 16),
                                       nn.ReLU(),
                                       nn.Linear(16, 1)).to(self.device)

        # self.value_net[-1].weight.data = torch.abs(self.value_net[-1].weight.data)
        # self.value_net[-1].bias.data = torch.abs(self.value_net[-1].bias.data)

        self.Wk = nn.Linear(self.fl_repr_size, 1, bias=False).to(self.device)
        self.Wq = nn.Linear(self.fl_repr_size, 1, bias=False).to(self.device)
        self.Wv = nn.Linear(self.fl_repr_size, 1, bias=False).to(self.device)

        self.WkT = nn.Linear(self.fl_repr_size, 1, bias=False).to(self.device)
        self.WqT = nn.Linear(self.fl_repr_size, 1, bias=False).to(self.device)
        self.WvT = nn.Linear(self.fl_repr_size, 1, bias=False).to(self.device)

        params = list(self.schedule_embedding.parameters()) + list(self.attention_ff.parameters()) \
                 + list(self.action_attention_ff.parameters()) + list(self.action_embedding.parameters()) \
                 + list(self.Wk.parameters()) + list(self.Wq.parameters()) + list(self.Wv.parameters()) \
                 + list(self.WkT.parameters()) + list(self.WqT.parameters()) + list(self.WvT.parameters()) \
                 + list(self.value_net.parameters()) \

        self.optimizer = optim.Adam(params, weight_decay=weight_decay, lr=l_rate)

    def forward(self, state, actions, mask, num_flights, num_airlines, reward):
        state = state.reshape((-1, state.shape[-1]))
        actions = actions.reshape((-1, actions.shape[-1]))  # /302200
        schedule_len = (num_airlines+self.flightDiscretisation)*num_flights



        schedules = state[:, : schedule_len]
        current_trade = state[:, -num_flights:]

        schedules = schedules.reshape((schedules.shape[0], num_flights, self.flightDiscretisation + num_airlines))
        schedules = schedules[:, :, num_airlines:]

        schedules_with_current = torch.cat([schedules, current_trade.unsqueeze(-1)], dim=-1)
        embedded = self.schedule_embedding(schedules_with_current)
        mask_non_zero = torch.nonzero(mask - 1)
        if mask_non_zero.shape[0] > 0:
            if mask_non_zero.shape[1] > 1:
                embedded[mask_non_zero[0], mask_non_zero[1], :] = 0
            else:
                mask_non_zero = mask_non_zero.squeeze(-1)
                embedded[0, mask_non_zero, :] = 0
        k = self.Wk(embedded)
        q = self.Wq(embedded)
        v = self.Wv(embedded)

        attention_matrix = sMax(torch.matmul(k, q.transpose(1, 2)))
        self_attention = torch.matmul(attention_matrix, v).transpose(1, 2)

        add_norm = self_attention.transpose(1, 2) + embedded

        add_norm = nn.functional.normalize(add_norm, p=2, dim=-1)

        second_add_norm = self.attention_ff(add_norm)
        second_add_norm = second_add_norm + add_norm

        second_add_norm = nn.functional.normalize(second_add_norm, p=2, dim=-1)

        k = self.WkT(second_add_norm)
        v = self.WvT(second_add_norm)

        actions = actions[:, num_airlines:]
        embedded_actions = self.action_embedding(actions)
        q = self.WqT(embedded_actions).unsqueeze(-2)
        action_attention = sMax(torch.matmul(q.transpose(1, 2), k.transpose(1, 2)))

        action_self_attention = torch.matmul(action_attention, v).transpose(1, 2).squeeze(-2)

        add_norm = action_self_attention + embedded_actions
        add_norm = nn.functional.normalize(add_norm, p=2, dim=-1)

        post_attention = self.action_attention_ff(add_norm)

        post_attention += add_norm
        post_attention = nn.functional.normalize(post_attention, p=2, dim=-1)

        result = self.value_net(post_attention)

        return result + reward.unsqueeze(-1)

    def pick_action(self, state, action, mask, num_flights, num_airlines, reward):
        reward = torch.tensor(reward).to(self.device)
        with torch.no_grad():
            scores = self.forward(state.to(self.device), action.to(self.device), mask.to(self.device),
                                  num_flights, num_airlines, reward)
        return scores

    def update_weights(self, batch: tuple, gamma: float = 1.0):
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
        self.loss = self.loss * 0.9 + 0.1 * loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        if self.loss < self.bestLoss:
            torch.save(self.state_dict(), "air.pt")

        return loss

    def update_weights_episode(self, batch: tuple, gamma: float = 1.0):
        criterion = torch.nn.MSELoss()

        states, actions, partial_rewards, rewards, masks = (element.to(self.device) for element in batch[:-2])
        num_flights, num_airlines = batch[-2], batch[-1]
        actions_tensor = states[:, :(self.flightDiscretisation + num_airlines) * num_flights]
        actions_tensor = actions_tensor.reshape((actions_tensor.shape[0], num_flights,
                                                 self.flightDiscretisation + num_airlines))

        actions_tensor = actions_tensor[torch.nonzero(actions, as_tuple=True)]
        loss = 0
        for i in range(100):
            self.zero_grad()
            Q = self.forward(states, actions_tensor, masks,num_flights, num_airlines, partial_rewards)
            rewards = rewards.reshape((rewards.shape[0], -1))
            loss = criterion(Q, rewards)
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optimizer.step()

            self.loss = loss.item()

        # if self.loss < self.bestLoss:
        #     torch.save(self.state_dict(), "air.pt")

        return loss

        # schedules = schedules.reshape((schedules.shape[0], -1, self.singleFlightSize)).to(self.device)
        # trades = trades.reshape((trades.shape[0], -1, self.tradeSize)).to(self.device)
        #
        # schedules_e = self.schedule_embedding(schedules).to(self.device)
        # trades_e = self.trade_embedding(trades).to(self.device)
        # current_trades_e = self.trade_embedding(current_trades).to(self.device)
        #
        # trades_w = torch.matmul(trades_e, current_trades_e.unsqueeze(2)).to(self.device)
        # trades_w = sMax(trades_w)
        # trades_attention = torch.matmul(trades_e.transpose(1, 2), trades_w).to(self.device)
        #
        # schedules_w = torch.matmul(schedules_e, trades_attention).to(self.device)
        # schedules_w = sMax(schedules_w)
        # schedules_attention = torch.matmul(schedules_e.transpose(1, 2), schedules_w).to(self.device)

        # value_in = torch.cat((schedules_attention, trades_attention), dim=1).transpose(1, 2).squeeze(1).to(self.device)
        # value_in = torch.cat((value_in, current_trades_e), dim = 1).to(self.device)
