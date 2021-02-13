import trainer
import masker
from Agents import hyperAgent
import instanceMaker
import pandas as pd
from ModelStructure.Costs.costFunctionDict import CostFuns

# schedule_tensor[i, self.airDict[self.flights[i].airline.name]] = 1
# schedule_tensor[i, self.numAirlines + self.flightTypeDict[self.flights[i].type]] = 1
# schedule_tensor[i, -2] = self.flights[i].slot.time / self.reductionFactor
# schedule_tensor[i, -1] = self.flights[i].eta / self.reductionFactor


# problem's parameters
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
print("the solution should be:\n", [[tuple(pair[0]), tuple(pair[1])] for pair in instance.matches])

# hyper agent parameters
weight_decay = 1e-5
batch_size = 100
memory_size = 20_000

trainings_per_step = 5

hyper_agent = hyperAgent.HyperAgent(num_flight_types, num_airlines, num_flights, num_trades, num_combs,
                                    trainings_per_step=trainings_per_step,
                                    weight_decay=weight_decay, batch_size=batch_size,
                                    memory_size=memory_size, train_mode=True)

# trainer parameters
EPS_DECAY: float = 1000
eps_fun = lambda i, num_iterations: max(0.05, 1 - i / 10_000)  # np.exp(- 4*i/num_iterations)

train = trainer.Trainer(hyper_agent, length_episode=num_trades, eps_fun=eps_fun, eps_decay=EPS_DECAY)
train.run(500_000, df, training_start_iteration=100)

# print(train.episode(instance.get_schedule_tensor()))
