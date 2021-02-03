from typing import List

import torch


class Trainer:

    def __init__(self, hyper_agent, length_episode):
        self.hyperAgent = hyper_agent
        self.LengthEpisode = length_episode

    def step(self, state_list: List):
        return self.hyperAgent.get_trade(state_list)

    def episode(self,schedule_tensor: torch.tensor):
        trade_list = []
        for i in range(self.LengthEpisode):
            trade_list.append(self.step([schedule_tensor, trade_list]))
        return trade_list

