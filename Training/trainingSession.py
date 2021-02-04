import trainer
import flAgent
import airAgent
import hyperAgent
import instanceMaker
import pandas as pd

df = pd.read_csv("custom_5_5.csv")
instance = instanceMaker.Instance(triples=False, df=df)
instance.run()
instance.print_performance()
print(instance.matches)

num_trades = 12
num_flights = 20

input_size = 21 * num_flights + (num_trades+1)*28

AIR = airAgent.AirNet(input_size, num_flights= None, num_airlines=4, num_trades= None)
FL = flAgent.FlNet(input_size, None, None, None, couples_combs=10)
hyper_agent = hyperAgent.HyperAgent(AIR, FL)

train = trainer.Trainer(AIR, FL, hyper_agent, length_episode=num_trades)
train.run(10000, df)

# print(train.episode(instance.get_schedule_tensor()))





