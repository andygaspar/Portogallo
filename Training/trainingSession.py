import torch

import trainer
import masker
from Agents import hyperAttentiveAgent
import instanceMaker
import pandas as pd
from ModelStructure.Costs.costFunctionDict import CostFuns

# problem's parameters
# from Training.noneMasker import NoneMasker

DISCRETISATION_SIZE = 50

num_flight_types = len(CostFuns().flightTypeDict)
num_trades = 2
num_airlines = 4
num_flights = 20

# fixed particular instance (copied inside the trainer - trainer must be changed in the future)
df = pd.read_csv("custom_5_5.csv")
instance = instanceMaker.Instance(triples=True, df=df)
instance.run()

"""
FD0 0
FA1 1
FD2 2
FB6 3
FC4 4
FD5 5
FB10 6
FC7 7
FA8 8
FA9 9
FB13 10
FC11 11
FB12 12
FB3 13
FD14 14
FC19 15
FA17 16
FA16 17
FD18 18
FC15 19
"""


print(instance.airlines)
print("\n\n\n\n")
instance.print_performance()
print(instance.matches[0])
print("all feasible matches:\n", [[tuple(pair[0]), tuple(pair[1])] for pair in instance.matches])
print("solution:", instance.offers_selected)

# hyper agent parameters
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
MEMORY_SIZE = 200

hyper_agent = hyperAttentiveAgent.AttentiveHyperAgent(num_airlines, num_flights, num_trades,
                                                      weight_decay=WEIGHT_DECAY, l_rate=LEARNING_RATE,
                                                      batch_size=BATCH_SIZE, discretisation_size=DISCRETISATION_SIZE,
                                                      memory_size=MEMORY_SIZE, train_mode=True)


# trainer parameters
START_TRAINING = 1
EPS_DECAY: float = 1000
MIN_REWARD = -100000


#eps_fun = lambda i, num_iterations: max(0.05, 1 - i / 10_000)  # np.exp(- 4*i/num_iterations)
# eps_fun = lambda i, num_iterations: 0.1 if i > START_TRAINING else 1
eps_fun = lambda i, num_iterations: 1 - i/num_iterations

# masker = NoneMasker

train = trainer.Trainer(hyper_agent, length_episode=num_trades,
                        eps_fun=eps_fun, min_reward=MIN_REWARD,  eps_decay=EPS_DECAY, triples=True)
train.run(5000, df, training_start_iteration=START_TRAINING, train_t=10)


#
# for g in hyper_agent.AirAgent.optimizer.param_groups:
#     g['lr'] = 0.001
# for g in hyper_agent.FlAgent.optimizer.param_groups:
#     g['lr'] = 0.001
#
# train.run(2500, df, training_start_iteration=1000, train_t=200)
#
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
#
# # print(train.episode(instance.get_schedule_tensor()))
