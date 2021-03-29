import pandas as pd
import numpy as np
import torch

from ModelStructure.Costs.costFunctionDict import CostFuns
from Training.Agents.enconding import Encoder

df = pd.read_csv("replay.csv")
print(df)
data = torch.tensor(df.values[:, 1:], dtype=torch.float).to("cuda:0")[:1]
# dd, _, _ = torch.split(data,[21*16, 20*6, 20], dim=-1)
# dd = torch.split(dd, [21 for _ in range(16)], dim=-1)
# print(dd[:3])
print(data)
for i in range(16):
    data[0, i*21 + 19] *= 10
    data[0, i * 21 + 20] *= 10

print(data)

num_flight_types = len(CostFuns().flightTypeDict)
num_trades = 6
num_airlines = 4
num_combs = 6
num_flights = 16

ETA_info_size = 1
time_info_size = 1
singleTradeSize = (num_airlines + num_combs) * 2  # 2 as we are dealing with couples
currentTradeSize = singleTradeSize
numCombs = num_combs
numAirlines = num_airlines

input_size = (num_flight_types + ETA_info_size + time_info_size + num_airlines) * num_flights + \
             num_trades * singleTradeSize + currentTradeSize

lr = 1e-4
weight_decay = 1e-4
output_size = 3

sample_size = 1

enc = Encoder(input_size, lr, weight_decay, num_flight_types, num_airlines, num_flights, num_trades, num_combs, output_size)

for j in range(10000):
    for i in range(50):
        sample_idxs = np.random.choice(data.shape[0], sample_size)
        enc.update_weights(data[sample_idxs])
    print(j*100, enc.loss)
    if j%50 == 0:
        # print(enc.forward(data[230])[21*4:21*5], data[230][21*4:21*5])
        # print(enc.forward(data[1001])[21 * 4:21 * 5], data[1001][21 * 4:21 * 5])
        for i in range(16):
            print(enc.forward(data[0])[21 * i:21 * (i+1)], data[0][21 * i:21 * (i+1)])




