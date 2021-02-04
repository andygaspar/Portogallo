import trainer

import hyperAgent
import instanceMaker
import pandas as pd
from ModelStructure.Costs.costFunctionDict import CostFuns

# schedule_tensor[i, self.airDict[self.flights[i].airline.name]] = 1
# schedule_tensor[i, self.numAirlines + self.flightTypeDict[self.flights[i].type]] = 1
# schedule_tensor[i, -2] = self.flights[i].slot.time / self.reductionFactor
# schedule_tensor[i, -1] = self.flights[i].eta / self.reductionFactor


# problem's parameters
num_flight_types = len(CostFuns().flightTypeDict)
num_trades = 12
num_airlines = 4
num_combs = 10
num_flights = 20

# fixed particular instance (copied inside the trainer - trainer must be changed in the future)
df = pd.read_csv("custom_5_5.csv")
instance = instanceMaker.Instance(triples=False, df=df)
instance.run()
instance.print_performance()
print(instance.matches[0])
print("the solution should be:\n", [[tuple(pair[0]), tuple(pair[1])] for pair in instance.matches])

# hyper agent parameters
weight_decay = 1e-5
batch_size = 100
memory_size = 10000

hyper_agent = hyperAgent.HyperAgent(num_flight_types, num_airlines, num_flights, num_trades, num_combs,
                                    weight_decay=weight_decay, batch_size=batch_size,
                                    memory_size=memory_size, train_mode=False)

train = trainer.Trainer(hyper_agent, length_episode=num_trades)
train.run(10000, df)

# print(train.episode(instance.get_schedule_tensor()))
