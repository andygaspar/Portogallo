import torch

import trainer
import masker
from Agents import hyperAgent
from Agents import hyperAttentiveAgent
import instanceMaker
import pandas as pd
from ModelStructure.Costs.costFunctionDict import CostFuns

# schedule_tensor[i, self.airDict[self.flights[i].airline.name]] = 1
# schedule_tensor[i, self.numAirlines + self.flightTypeDict[self.flights[i].type]] = 1
# schedule_tensor[i, -2] = self.flights[i].slot.time / self.reductionFactor
# schedule_tensor[i, -1] = self.flights[i].eta / self.reductionFactor


# problem's parameters
from Training.noneMasker import NoneMasker
import random

random.seed(0)
torch.manual_seed(0)

num_flight_types = len(CostFuns().flightTypeDict)
num_trades = 6
num_airlines = 4
num_combs = 6
num_flights = 16

# fixed particular instance (copied inside the trainer - trainer must be changed in the future)
df = pd.read_csv("custom_4_4.csv")
instance = instanceMaker.Instance(triples=False, df=df)
instance.run()

print(instance.airlines)
print("\n\n\n\n")
instance.print_performance()
print(instance.matches[0])
print("all feasible matches:\n", [[tuple(pair[0]), tuple(pair[1])] for pair in instance.matches])
print("solution:", instance.offers_selected)

# hyper agent parameters
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024
MEMORY_SIZE = 200_000

hyper_agent = hyperAttentiveAgent.AttentiveHyperAgent(num_flight_types, num_airlines, num_flights, num_trades, num_combs,
                                                      weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                                                      memory_size=MEMORY_SIZE, train_mode=True)
#hyper_agent = hyperAgent.HyperAgent(num_flight_types, num_airlines, num_flights, num_trades, num_combs,
#                                    weight_decay=weight_decay, batch_size=batch_size,
#                                    memory_size=memory_size, train_mode=True)


# trainer parameters
START_TRAINING = 20
EPS_DECAY: float = 1000
MIN_REWARD = -100
ITERATIONS = 2500


eps_fun = lambda i, num_iterations: max(0.05, 1 - i / ITERATIONS)  # np.exp(- 4*i/num_iterations)
# eps_fun = lambda i, num_iterations: 0.1 if i > START_TRAINING else 1

# masker = NoneMasker

train = trainer.Trainer(hyper_agent, length_episode=num_trades,
                        eps_fun=eps_fun, min_reward=MIN_REWARD,  eps_decay=EPS_DECAY)
train.run(ITERATIONS, df, training_start_iteration=START_TRAINING, train_t=10)

for g in hyper_agent.AirAgent.optimizer.param_groups:
    g['lr'] = 0.001
for g in hyper_agent.FlAgent.optimizer.param_groups:
    g['lr'] = 0.001

train.run(2500, df, training_start_iteration=1000, train_t=200)

# for g in hyper_agent.AirAgent.optimizer.param_groups:
#     g['lr'] = 0.00001
# for g in hyper_agent.FlAgent.optimizer.param_groups:
#     g['lr'] = 0.00001
#
# train.run(2500, df, training_start_iteration=1000, train_t=200)
#
#
# replay = hyper_agent.AirReplayMemory
# replay_df = pd.DataFrame(replay.states.numpy())
# replay_df.to_csv("replay.csv")

# print(train.episode(instance.get_schedule_tensor()))
