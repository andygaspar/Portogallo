from ctypes import cdll, pointer
import ctypes
import numpy as np
from RL import rl
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
import time
from itertools import permutations
import copy

# load the library
lib = cdll.LoadLibrary('./liboffers.so')

np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 20
num_airlines = 3

schedule_df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[0])
cost_fun = CostFuns().costFun["realistic"]

# create model
rl_model = rl.Rl(schedule_df, cost_fun)


"""

t = time.perf_counter()
couple_matches = rl_model.get_couple_matches()
print("time to get all couple matches: ", time.perf_counter() - t)
print(len(couple_matches), " convenient pairs found")

t = time.perf_counter()
triple_matches = rl_model.get_triple_matches()
print("time to get all triple matches: ", time.perf_counter() - t)
print(len(triple_matches), "convenient triples found ")

# get an airline
airline = rl_model.airlines[0]
print(airline)

# get a flight
flight = airline.flights[0]
print(flight)

# get a couple of flights of an airline
couple = airline.flight_pairs[0]
print("the couple of flight tried to mathced is:", couple)
t = time.perf_counter()
couple_matches_for_flight = rl_model.check_couple_in_pairs(couple)
print("time to get all flight's couple matches: ", time.perf_counter() - t)

# get a couple of flights of an airline
couple = airline.flight_pairs[0]
t = time.perf_counter()
triple_matches_for_flight = rl_model.check_couple_in_triples(couple)
print("time to get all flight's triple matches: ", time.perf_counter() - t)
"""

couples = list(permutations([0, 1, 2, 3]))
couples_copy = copy.copy(couples)
for c in couples_copy:
    if (c[0] == 0 and c[1] == 1) or (c[0] == 1 and c[1] == 0):
        couples.remove(c)
couples = np.array(couples).astype(np.short)

triples = list(permutations([0, 1, 2, 3, 4, 5]))
triples_copy = copy.copy(triples)
for t in triples_copy:
    if ((t[0] == 0 and t[1] == 1) or (t[0] == 1 and t[1] == 0)) or \
            ((t[2] == 2 and t[3] == 3) or (t[2] == 3 and t[3] == 2)) or \
            ((t[4] == 4 and t[5] == 5) or (t[4] == 5 and t[5] == 4)):
        triples.remove(t)
triples = np.array(triples).astype(np.short)


# create a Geek class
class OfferChecker(object):

    # constructor
    def __init__(self, vect, coup, trip):
        # attribute
        # lib.OfferChecker_.argtypes = [ctypes.c_void_p, ctypes.c_short, ctypes.c_short, ctypes.c_void_p, ctypes.c_short,
        #                               ctypes.c_short, ctypes.c_void_p, ctypes.c_short, ctypes.c_short]
        lib.OfferChecker_.restype = ctypes.c_void_p
        self.obj = lib.OfferChecker_(vect.ctypes.data, vect.shape[0], vect.shape[1],
                                     coup.ctypes.data, coup.shape[0], coup.shape[1],
                                     trip.ctypes.data, trip.shape[0], trip.shape[1])

    def check(self, flights):
        fl = np.array(flights).astype(np.short)
        ciccio = np.array(flights).astype(np.short).ctypes.data

        return lib.check_couple_condition_(self.obj, ciccio)

    def air_couple_check(self, fl_pair_a, fl_pair_b):
        air_pairs = []
        for pairA in fl_pair_a:
            for pairB in fl_pair_b:
                p= [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB]
                air_pairs += [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB]
        # print("combs", len(fl_pair_a)*len(fl_pair_b), len(air_pairs))
        lib.air_couple_check_.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        pappo = np.array(air_pairs).astype(np.short)
        ciccio = pappo.ctypes.data
        return lib.air_couple_check_(self.obj,ciccio)

    def print_mat(self):
        lib.print_mat_(self.obj)

    def print_couples(self):
        lib.print_couples_(self.obj)

    def print_triples(self):
        lib.print_triples_(self.obj)

    # create a Geek class object


# print(rl_model.scheduleMatrix.dtype)
# print(rl_model.scheduleMatrix,"\n\n")
f = OfferChecker(rl_model.scheduleMatrix, couples, triples)

# f.print_mat()
# f.print_couples()
# f.print_triples()
fl_pair_a = rl_model.airlines_pairs[0][0].flight_pairs
fl_pair_b = rl_model.airlines_pairs[0][1].flight_pairs
import checkOffers

print(len(checkOffers.air_couple_check(rl_model.scheduleMatrix, rl_model.airlines_pairs[0])))
print(f.air_couple_check(fl_pair_a, fl_pair_b))

# print(f.check(
#     [fl.slot.index for fl in airline.flight_pairs[0]] + [fl.slot.index for fl in rl_model.airlines[2].flight_pairs[0]]))
