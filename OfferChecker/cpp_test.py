import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
from Training import instanceMaker
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
import time
from itertools import permutations
import copy
import os
from OfferChecker.checkOffer import OfferChecker as O



lib = ctypes.CDLL('./liboffers.so')
lib.OfferChecker_.argtypes = [ctypes.c_void_p, ctypes.c_short, ctypes.c_short, ctypes.c_void_p, ctypes.c_short,
                              ctypes.c_short, ctypes.c_void_p, ctypes.c_short, ctypes.c_short]
lib.air_couple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
lib.air_triple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
# load the library
# lib = cdll.LoadLibrary('./liboffers.so')

np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 100
num_airlines = 10

schedule_df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[0])
cost_fun = CostFuns().costFun["realistic"]

# create model
rl_model = instanceMaker.Instance(schedule_df, cost_fun)

print("flights: ", num_flights, "   num airlines: ", num_airlines)


# t = time.perf_counter()
# couple_matches = rl_model.all_couples_matches()
# print("n couples: ", len(couple_matches), "   python time: ", time.perf_counter() - t)

# print(couple_matches)

# t = time.perf_counter()
# triple_matches = rl_model.all_triples_matches()
# print("n triples: ", len(triple_matches), "    python time triples:", time.perf_counter() - t)

# get an airline
airline = rl_model.airlines[0]


# get a couple of flights of an airline
# couple = airline.flight_pairs[0]
# print("the couple of flight tried to mathced is:", couple)
# t = time.perf_counter()
# couple_matches_for_flight = rl_model.check_couple_in_pairs(couple)
# print("time to get all flight's couple matches: ", time.perf_counter() - t)
#
# get a couple of flights of an airline
couple = airline.flight_pairs[0]
# t = time.perf_counter()
# triple_matches_for_flight = rl_model.check_couple_in_triples(couple)
# print("time to get all flight's triple matches: ", time.perf_counter() - t)


couples = list(permutations([0, 1, 2, 3]))
couples_copy = copy.copy(couples)
for c in couples_copy:
    if (c[0] == 0 and c[1] == 1) or (c[0] == 1 and c[1] == 0):
        couples.remove(c)
couples = np.array(couples).astype(np.short)
# print(len(couples))

triples = list(permutations([0, 1, 2, 3, 4, 5]))
triples_copy = copy.copy(triples)
for t in triples_copy:
    if ((t[0] == 0 and t[1] == 1) or (t[0] == 1 and t[1] == 0)) or \
            ((t[2] == 2 and t[3] == 3) or (t[2] == 3 and t[3] == 2)) or \
            ((t[4] == 4 and t[5] == 5) or (t[4] == 5 and t[5] == 4)):
        triples.remove(t)
triples = np.array(triples).astype(np.short)
# print(len(triples))


# create a Geek class
class OfferChecker(object):

    # constructor
    def __init__(self, vect, coup, trip):
        # attribute
        lib.OfferChecker_.restype = ctypes.c_void_p
        self.obj = lib.OfferChecker_(ctypes.c_void_p(vect.ctypes.data), ctypes.c_short(vect.shape[0]), ctypes.c_short(vect.shape[1]),
                                     ctypes.c_void_p(coup.ctypes.data), ctypes.c_short(coup.shape[0]), ctypes.c_short(coup.shape[1]),
                                     ctypes.c_void_p(trip.ctypes.data), ctypes.c_short(trip.shape[0]), ctypes.c_short(trip.shape[1]))

    def check(self, flights):
        fl = np.array(flights).astype(np.short)
        ciccio = np.array(flights).astype(np.short).ctypes.data

        return lib.check_couple_condition_(self.obj, ciccio)

    def air_couple_check(self, fl_pair_a, fl_pair_b):
        air_pairs = []
        input_vect = []
        for pairA in fl_pair_a:
            for pairB in fl_pair_b:
                air_pairs.append([pairA, pairB])
                input_vect += [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB]

        len_array = len(fl_pair_a) * len(fl_pair_b)

        lib.air_couple_check_.restype = ndpointer(dtype=ctypes.c_bool, shape=(len_array,))
        input_vect = np.array(input_vect).astype(np.short)
        answer = lib.air_couple_check_(ctypes.c_void_p(self.obj),
                                       ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))

        return [air_pairs[i] for i in range(len_array) if answer[i]]

    def air_triple_check(self, fl_pair_a, fl_pair_b, fl_pair_c):
        air_trips = []
        input_vect = []
        for pairA in fl_pair_a:
            for pairB in fl_pair_b:
                for pairC in fl_pair_c:
                    air_trips.append([pairA, pairB, pairC])
                    input_vect += [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB] + \
                                  [fl.slot.index for fl in pairC]

        len_array = len(fl_pair_a) * len(fl_pair_b) * len(fl_pair_c)

        lib.air_triple_check_.restype = ndpointer(dtype=ctypes.c_bool, shape=(len_array,))
        input_vect = np.array(input_vect).astype(np.short)

        answer = lib.air_triple_check_(ctypes.c_void_p(self.obj),
                                       ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))
        return [air_trips[i] for i in range(len_array) if answer[i]]

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
ob = O(rl_model.scheduleMatrix)
# f.print_mat()
# f.print_couples()
# f.print_triples()

#
# coup = []
# # print(len(checkOffers.air_couple_check(rl_model.scheduleMatrix, rl_model.airlines_pairs[0])))
# t = time.perf_counter()
# for air_pair in rl_model.airlines_pairs:
#     fl_pair_a = air_pair[0].flight_pairs
#     fl_pair_b = air_pair[1].flight_pairs
#     coup += f.air_couple_check(fl_pair_a, fl_pair_b)
#
# print("n couples: ", len(coup), "   C++ time: ", time.perf_counter() - t)
#
# t = time.perf_counter()
# cop = ob.all_couples_check(rl_model.airlines_pairs)
# print("n couples: ", len(cop), "   C++ obj time: ", time.perf_counter() - t)
#
#
# #print(coup)
#
trip = []
t = time.perf_counter()
for air_pair in rl_model.airlines_triples:
    fl_pair_a = air_pair[0].flight_pairs
    fl_pair_b = air_pair[1].flight_pairs
    fl_pair_c = air_pair[2].flight_pairs
    trip += f.air_triple_check(fl_pair_a, fl_pair_b, fl_pair_c)

print("n triples: ", len(trip), "   C++ time: ", time.perf_counter() - t)


t = time.perf_counter()
tri = ob.all_triples_check(rl_model.airlines_triples)
print("n triples: ", len(tri), "   C++ obj time: ", time.perf_counter() - t)
print(trip)

# print(f.check(
#     [fl.slot.index for fl in airline.flight_pairs[0]] + [fl.slot.index for fl in rl_model.airlines[2].flight_pairs[0]]))



print("the couple of flight tried to mathced is:", couple)
t = time.perf_counter()
cmff = ob.check_couple_in_pairs(couple, rl_model.airlines_pairs)
print("time to get all flight's couple matches: ", time.perf_counter() - t)
#
# get a couple of flights of an airline
t = time.perf_counter()
tmff = ob.check_couple_in_pairs(couple, rl_model.airlines_triples)
print("time to get all flight's triple matches: ", time.perf_counter() - t)