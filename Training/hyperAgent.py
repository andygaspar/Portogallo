import copy

import torch


class HyperAgent:

    def __init__(self, air_agent, fl_agent):
        self.FlAgent = fl_agent
        self.AirAgent = air_agent
        self.currentTrade = torch.zeros(28)

    def get_trade(self, state_list, pair=True):
        if pair:
            air_states = [torch.zeros(28)]
            self.currentTrade[0:4] = self.pick_action(self.AirAgent(state_list, self.currentTrade))
            fl_states = [copy.deepcopy(self.currentTrade)]
            self.currentTrade[4:14] = self.pick_action(self.FlAgent(state_list, self.currentTrade))
            air_states.append(copy.deepcopy(self.currentTrade))
            self.currentTrade[14:18] = self.pick_action(self.AirAgent(state_list, self.currentTrade))
            fl_states.append(copy.deepcopy(self.currentTrade))
            self.currentTrade[18:28] = self.pick_action(self.FlAgent(state_list, self.currentTrade))

            return self.currentTrade, air_states, fl_states

    def pick_action(self, scores):
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action
