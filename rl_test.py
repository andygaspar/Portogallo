from Istop import istop
from NoNegativeBound import nnBound
from Training import instanceMaker
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.Slot.slot import Slot
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from Istop.AirlineAndFlight.istopFlight import IstopFlight
import numpy as np
import time

airline: IstopAirline
flight: IstopFlight


#np.random.seed(0)
scheduleTypes = scheduleMaker.schedule_types(show=True)

num_airlines = 5
num_flights = 35

instance = instanceMaker.Instance(num_flights=num_flights, num_airlines=num_airlines)
mat = instance.get_schedule_tensor()

print(instance.airDict)
print(instance.flightDict)

# for i in range(mat.shape[0]):
#     flight = instance.flights[i]
#     print(flight.airline, flight.type, instance.flightDict[flight.type], flight, flight.slot.time - flight.eta)
#     print(mat[i])

from Old_RL import old_rl
import pandas as pd

df = scheduleMaker.df_maker(custom = [5, 5, 5, 5])
mod = old_rl.Rl(df, CostFuns().costFun["realistic"])
dfSchedules = mod.udppDf
dfSchedules["instance"] = [0 for _ in range(20)]
dfSchedMat = pd.DataFrame(mod.scheduleMatrix.T)
dfSchedMat["instance"] = [0 for _ in range(22)]

for i in range(1, 201):
    df = scheduleMaker.df_maker(custom=[5, 5, 5, 5])
    mod = old_rl.Rl(df, CostFuns().costFun["realistic"])
    dfS = mod.udppDf
    dfS["instance"] = [i for _ in range(20)]
    dfSM = pd.DataFrame(mod.scheduleMatrix.T)
    dfSM["instance"] = [i for _ in range(22)]

    dfSchedules = dfSchedules.append(dfS, ignore_index=True)
    dfSchedMat = dfSchedMat.append(dfSM, ignore_index=True)

dfSchedules.to_csv("CSV/schedules.csv")
dfSchedMat.to_csv("CSV/schedMat.csv")