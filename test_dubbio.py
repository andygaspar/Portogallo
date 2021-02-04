
from Istop import istop
from NoNegativeBound import nnBound
from Old_RL import old_rl
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.Slot.slot import Slot
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from Istop.AirlineAndFlight.istopFlight import IstopFlight
import numpy as np
import time
import pandas as pd

airline: IstopAirline
flight: IstopFlight

np.random.seed(0)
scheduleTypes = scheduleMaker.schedule_types(show=True)

# init variables, chedule and cost function
num_flights = 70
num_airlines = 10
# schedule_df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleTypes[0])



schedule_df = scheduleMaker.df_maker(custom=[5,5,5,5])
schedule_df.to_csv("custom_5_4.csv")
#schedule_df_1 = pd.read_csv("custom_5_5.csv")
#print(schedule_df["type"]==schedule_df_1["type"])
cost_fun = CostFuns().costFun["realistic"]


# create model
# rl_model = old_rl.Rl(schedule_df, cost_fun, triples=True, parallel=False)
#
#
# t = time.perf_counter()
# couple_matches = rl_model.all_couples_matches()
# print("time to get all couple matches: ", time.perf_counter() - t)
# print(len(couple_matches), " convenient pairs found", "\n")
#
# t = time.perf_counter()
# triple_matches = rl_model.all_triples_matches()
# print("time to get all triple matches: ", time.perf_counter() - t)
# print(len(triple_matches), "convenient triples found ", "\n")
#
#
rl_model = old_rl.Rl(schedule_df, cost_fun, triples=False, parallel=True, private=False)
#
#
t = time.perf_counter()
couple_matches = rl_model.all_couples_matches()
print("time to get all couple matches: ", time.perf_counter() - t)
print(len(couple_matches), " convenient pairs found", "\n")
#
# t = time.perf_counter()
# triple_matches = rl_model.all_triples_matches()
# print("time to get all triple matches: ", time.perf_counter() - t)
# print(len(triple_matches), "convenient triples found ", "\n")
#
# t = time.perf_counter()
# triple_matches = rl_model.all_triples_matches_fast()
# print("time to get all triple matches: ", time.perf_counter() - t)
# print(len(triple_matches), "convenient triples found ", "\n")


# rl_model = old_rl.Rl(schedule_df, cost_fun, triples=True, parallel=True, private=True)
t = time.perf_counter()
triple_matches = rl_model.all_triples_matches_fast()
print("time to get all triple matches: ", time.perf_counter() - t)
print(len(triple_matches), "convenient triples found ", "\n")

rl_model.run()
rl_model.print_performance()
print(rl_model.offers_selected)


print(rl_model.flightTypeDict)

#
# # get all airlines pairs and triples
# print("airline pairs ", rl_model.airlines_pairs, "\n")
# print("airline triples ", rl_model.airlines_triples, "\n")
#
# # get an airline
# airline = rl_model.airlines[1]
# print("airline", airline)
#
# # get a flight from the schedule
# flight = rl_model.flights[0]
# print(flight, "flight type: ", flight.type, "flight's slot: ", flight.slot.index, "\n")
# # or from a particular airline
# flight = airline.flights[0]
# print(flight, "flight type: ", flight.type, "flight's slot: ", flight.slot.index, "\n")
#
# # flight type dict
# print(rl_model.flightTypeDict, "\n")
#
# # flight type in numbers
# print(flight, "flight type: ", flight.type, "  type in numbers", rl_model.flightTypeDict[flight.type], "\n")
#
# # get a couple of flights of an airline
# couple = airline.flight_pairs[0]
# print("the couple of flight tried to be matched is:", couple)
# t = time.perf_counter()
# couple_matches_for_flight = rl_model.check_couple_in_pairs(couple)
# print("couples matches for couple: ", couple_matches_for_flight)
# print("time to check pair matches: ", time.perf_counter() - t, "\n")
#
# # get a couple of flights of an airline
# t = time.perf_counter()
# triple_matches_for_flight = rl_model.check_couple_in_triples(couple)
# print("triples matches for couple: ", triple_matches_for_flight)
# print("time to check triple matches: ", time.perf_counter() - t)
#
# print("flights ordered without flights of airline", airline, rl_model.get_filtered_schedule(airline))
