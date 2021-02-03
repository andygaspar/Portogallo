from typing import List
from Training import instanceMaker

import torch


class Trainer:

    def __init__(self, hyper_agent, length_episode):
        self.hyperAgent = hyper_agent
        self.lengthEpisode = length_episode

    def step(self, state_list: List):
        x = self.hyperAgent.get_trade(state_list)
        return x

    def episode(self, schedule_tensor: torch.tensor):
        trade_list = torch.zeros(28 * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trade_list[i * 28: (i + 1) * 28] = self.step([schedule_tensor, trade_list])
        return trade_list

    def run(self, num_iterations, df=None):
        if df is not None:
            instance = instanceMaker.Instance(triples=False, df=df)
        else:
            #to do
            return

        for i in range(num_iterations):
            schedule = instance.get_schedule_tensor()
            self.episode(schedule)