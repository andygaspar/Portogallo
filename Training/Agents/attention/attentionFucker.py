import copy
from typing import List

import numpy as np
import torch
from torch import optim
from Training.Agents.attention.attentionCodec import AttentionCodec
from Training.Agents import attentionAgent
from Training.Agents.replayMemory import ReplayMemory
from Training.masker import Masker

sMax = torch.nn.Softmax(dim=-1)


class AttentiveHyperAgent:

    def __init__(self, num_airlines, num_flights, num_trades, discretisation_size, weight_decay, l_rate,
                 trainings_per_step=10, batch_size=200, memory_size=10000, train_mode=False):

        MAX_NUM_FLIGHTS = 200

        self.singleTradeSize = num_flights  # 2 as we are dealing with couples
        self.currentTradeSize = self.singleTradeSize
        self.numAirlines = num_airlines
        self.discretisationSize = discretisation_size
        self.singleFlightSize = num_airlines + num_flights

        self.numFlights = num_flights
        self.weightDecay = weight_decay

        hidden_dim = 64
        self._hidden_dim = hidden_dim
        self._context_dim = self._hidden_dim + 20*3
        self._codec = AttentionCodec(self.discretisationSize + self.numAirlines, self._hidden_dim, n_heads=8, n_attention_layers=6, context_dim=None)
        self.context = None
        self.actions_embeddings=None


        ##TRAINING DA VEDERE
        self.trainMode = train_mode
        self.trainingsPerStep = trainings_per_step
        self.batchSize = batch_size

        self.replayMemory = ReplayMemory(self.discretisationSize * MAX_NUM_FLIGHTS, self.discretisationSize,
                                         size=memory_size)
        ######################

    def pick_flight(self, state, masker: Masker, num_flights):
        actions = torch.zeros_like(masker.mask)
        actions_tensor = state[:(self.discretisationSize + self.numAirlines) * num_flights]
        actions_tensor = actions_tensor.reshape((num_flights, self.discretisationSize + self.numAirlines))

        probs = self._codec.get_action_probs(self.context, actions_tensor, masker.mask)
        if self.trainMode:
            action = np.random.choice(range(probs.shape[0]), p=probs.to_numpy())
        else:
            action = np.argmax(probs.to_numpy())
        actions[action] = 1
        masker.set_action(action.item())
        return actions

    def init_step(self, schedule, trade_list, current_trade):
        self.actions_embeddings = self._codec.encode(schedule)
        self.context = torch.zeros(1, self._context_dim)
        self.context[0, :self._hidden_dim] = torch.mean(self.actions_embeddings, dim=0)


    def step(self, schedule: torch.tensor, trade_list: torch.tensor, eps, instance,
             len_step, len_episode, masker=None, last_step=False, train=True):

        num_flights = instance.numFlights
        current_trade = torch.zeros(num_flights)
        state = torch.cat([schedule, trade_list, current_trade], dim=-1)
        masker.set_initial_mask()

        self.replayMemory.set_initial_state(state)

        for _ in range(len_step - 1):
            action = self.pick_flight(state, eps, masker, num_flights, len_episode)
            current_trade += action
            state[-num_flights:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=masker.mask, reward=0)

        action = self.pick_flight(state, eps, masker, num_flights, len_episode)
        current_trade += action
        state[-num_flights:] = current_trade

        if not last_step:
            state[-num_flights:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=masker.mask, reward=0)
            return current_trade
        else:
            last_state = torch.ones_like(state) * -1
            return current_trade, last_state, action

    def assign_end_episode_reward(self, last_state, action, mask, shared_reward, actions_in_episode):
        self.replayMemory.add_record(next_state=last_state, action=action, mask=mask,
                                     reward=shared_reward, actions_in_episode=actions_in_episode, final=True)

    def train(self):
        for i in range(self.trainingsPerStep):
            batch, idxs = self.replayMemory.sample(self.batchSize)
            loss = self.network.update_weights(batch)
            self.replayMemory.update_losses(idxs, loss)

    def episode_training(self, num_flights, num_trades):
        batch = self.replayMemory.get_last_episode()
        self.network.update_weights_episode(batch, num_flights, num_trades)
