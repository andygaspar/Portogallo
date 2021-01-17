from ModelStructure.ScheduleMaker import scheduleMaker
from UDPP.udppModel import UDPPmodel
from ModelStructure.Costs.costFunctionDict import CostFuns
import pandas as pd

final_df: pd.DataFrame

costFun = CostFuns().costFun["step"]

final_df = pd.DataFrame(columns=["instance", "airline", "margins", "priority", "eta", "slot", "new slot"])
scheduleType = scheduleMaker.schedule_types(show=False)

for i in range(5000):
    df = scheduleMaker.df_maker(50, 4, distribution=scheduleType[0])
    udMod = UDPPmodel(df, costFun)
    udMod.run(optimised=True)
    for airline in udMod.airlines:
        for flight in airline.flights:
            to_append = [i, flight.airline, flight.margin, flight.priority, flight.eta, flight.slot.time,
                         flight.newSlot.time]
            a_series = pd.Series(to_append, index=final_df.columns)
            final_df = final_df.append(a_series, ignore_index=True)

    if i % 50 ==0:
        print(i)

# standardisation
for col in final_df.columns[2:-1]:
    final_df[col] = (final_df[col] - final_df[col].mean()) / final_df[col].std()

final_df.to_csv("50_4_increase.csv")
