from Istop import istop
from NoNegativeBound import nnBound
from RL import rl
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import time



np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 30
num_airlines = 4


schedule_df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[3])
cost_fun = CostFuns().costFun["realistic"]


#create model
rl_model = rl.Rl(schedule_df, cost_fun)


t = time.perf_counter()
couple_matches = rl_model.get_couple_matches()
print("time to get all couple matches: ", time.perf_counter()-t)
print(couple_matches)


t = time.perf_counter()
triple_matches = rl_model.get_triple_matches()
print("time to get all triple matches: ", time.perf_counter()-t)
print(triple_matches)


#get an airline
airline = rl_model.airlines[0]
print(airline)


#get a flight
flight = airline.flights[0]
print(flight)


#get a couple of flights of an airline
couple = airline.flight_pairs[0]
print("the couple of flight tried to mathced is:", couple)
t = time.perf_counter()
couple_matches_for_flight = rl_model.check_couple_in_pairs(couple)
print("time to get all flight's couple matches: ", time.perf_counter()-t)


#get a couple of flights of an airline
couple = airline.flight_pairs[0]
t = time.perf_counter()
triple_matches_for_flight = rl_model.check_couple_in_triples(couple)
print("time to get all flight's triple matches: ", time.perf_counter()-t)

