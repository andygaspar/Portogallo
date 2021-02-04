import trainer
import flAgent
import airAgent
import hyperAgent
import instanceMaker
import pandas as pd

df = pd.read_csv("custom_5_5.csv")

n_step= 12

input_size = 21*20 + (n_step+1)*28

AIR = airAgent.AirNet(input_size, num_flights= None, num_airlines=4, num_trades= None)
FL = flAgent.FlNet(input_size, None, None, None, couples_combs=10)
hyper_agent = hyperAgent.HyperAgent(AIR, FL)

train = trainer.Trainer(AIR, FL, n_step)
train.run(2,df)
instance = instanceMaker.Instance(triples=False, df=df)

# print(train.episode(instance.get_schedule_tensor()))

instance.run()
print(train.AirReplayMemory.states)
# instance.print_performance()



