import trainer
import flAgent
import airAgent
import hyperAgent
import instanceMaker

n_step= 12

input_size = 21*20 + n_step*28

AIR = airAgent.AirNet(input_size, num_flights= None, num_airlines=4, num_trades= None)
FL = flAgent.FlNet(input_size, None, None, None, couples_combs=10)
hyper_agent = hyperAgent.HyperAgent(AIR, FL)

train = trainer.Trainer(hyper_agent, n_step)

instance = instanceMaker(instanceMaker)


