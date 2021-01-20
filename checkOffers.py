import numpy as np
import copy
import concurrent.futures
from multiprocessing import Pool, RawArray, Array
from itertools import combinations, permutations
import time


def recursive_calls(flight, flights, free, initial_costs, airlines, comb):
    for other_flight in free:
        if other_flight != flight:
            if flight.eta <= other_flight.slot.time:
                fl = copy.copy(flights)
                fl.remove(flight)
                new_free = copy.copy(free)
                new_free.remove(other_flight)
                air = copy.copy(airlines)
                c = copy.deepcopy(comb)
                c[flight.airline.name].append([flight, other_flight.slot])
                if flight.airline.name != other_flight.airline.name:
                    air.append(flight.airline.name)

                still_convenient = True

                for airline in c.keys():
                    cost = sum([c[airline][i][0].costFun(c[airline][i][0], c[airline][i][1])
                                for i in range(len(c[airline]))])

                    if cost >= initial_costs[airline]:
                        still_convenient = False
                        break

                if still_convenient:
                    if len(fl) > 0:
                        if recursive_calls(fl[0], fl, new_free, initial_costs, air, c):
                            return True

                    elif len(np.unique(air)) == len(initial_costs.keys()):
                        return True
    return False


def get_combinations(flights, initial_costs):
    combs = dict(zip(np.unique([flight.airline.name for flight in flights]).tolist(),
                     [[] for i in range(int(len(flights) / 2))]))
    free = copy.copy(flights)
    return recursive_calls(flights[0], flights, free, initial_costs, [], combs)


def condition(pairs_list):
    airlines = [pair[0].airline.name for pair in pairs_list]
    initial_costs = dict(zip(airlines,
                             np.array(
                                 [pair[0].costFun(pair[0], pair[0].slot) + pair[1].costFun(pair[1], pair[1].slot)
                                  for
                                  pair in pairs_list])))

    flights = [flight for pair in pairs_list for flight in pair]
    return get_combinations(flights, initial_costs)


def check_couples(airl_pair):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:
            if condition([pairA, pairB]):
                matches.append([pairA, pairB])
    return matches


couples = list(permutations([0, 1, 2, 3]))
couples_copy = copy.copy(couples)
for c in couples_copy:
    if (c[0] == 0 and c[1] == 1) or (c[0] == 1 and c[1] == 0):
        couples.remove(c)
couples = np.array(couples)


triples = list(permutations([0, 1, 2, 3, 4, 5]))
triples_copy = copy.copy(triples)
for t in triples_copy:
    if ((t[0] == 0 and t[1] == 1) or (t[0] == 1 and t[1] == 0)) or \
            ((t[2] == 2 and t[3] == 3) or (t[2] == 3 and t[3] == 2)) or \
            ((t[4] == 4 and t[5] == 5) or (t[4] == 5 and t[5] == 4)):
        triples.remove(t)
triples = np.array(triples)


def check_couple_condition(mat, flights):
    for c in couples:
        # first airline eta check
        if mat[flights[0], 1] <= mat[flights[c[0]], 0]:
            if mat[flights[1], 1] <= mat[flights[c[1]], 0]:

                # check first airline's convenience
                if mat[flights[0], 2 + flights[0]] + mat[flights[1], 2 + flights[1]] > \
                        mat[flights[0], 2 + flights[c[0]]] + mat[flights[1], 2 + flights[c[1]]]:

                    # second airline eta check
                    if mat[flights[2], 1] <= mat[flights[c[2]], 0]:
                        if mat[flights[3], 1] <= mat[flights[c[3]], 0]:
                            if mat[flights[2], 2 + flights[2]] + mat[flights[3], 2 + flights[3]] > \
                                    mat[flights[2], 2 + flights[c[2]]] + \
                                    mat[flights[3], 2 + flights[c[3]]]:
                                return True
    return False


def air_couple_check(mat, airl_pair):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:

            if check_couple_condition(mat, [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB]):
                matches.append([pairA, pairB])
    return matches


def run_couples_check(mat, airlines_pairs, parallel=False):
    matches = []
    for air_pair in airlines_pairs:
        match = air_couple_check(mat, air_pair)
        if len(match) > 0:
            matches += match
    return matches







def check_triple_condition(mat, flights):
    matches = []
    for t in triples:

        # first airline eta check
        if mat[flights[0], 1] <= mat[flights[t[0]], 0]:
            if mat[flights[1], 1] <= mat[flights[t[1]], 0]:

                # check first airline's convenience
                if mat[flights[0], 2 + flights[0]] + mat[flights[1], 2 + flights[1]] > \
                        mat[flights[0], 2 + flights[t[0]]] + mat[flights[1], 2 + flights[t[1]]]:

                    # second airline eta check
                    if mat[flights[2], 1] <= mat[flights[t[2]], 0]:
                        if mat[flights[3], 1] <= mat[flights[t[3]], 0]:

                            # second convenience check
                            if mat[flights[2], 2 + flights[2]] + mat[flights[3], 2 + flights[3]] > \
                                    mat[flights[2], 2 + flights[t[2]]] + \
                                    mat[flights[3], 2 + flights[t[3]]]:

                                # third airline eta check
                                if mat[flights[4], 1] <= mat[flights[t[4]], 0]:
                                    if mat[flights[5], 1] <= mat[flights[t[5]], 0]:

                                        # third convenience check
                                        if mat[flights[4], 2 + flights[4]] + mat[flights[5], 2 + flights[5]] > \
                                                mat[flights[4], 2 + flights[t[4]]] + \
                                                mat[flights[5], 2 + flights[t[5]]]:
                                            return True
    return False


def air_triple_check(mat, airl_pair, parallel):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    fl_pair_c = airl_pair[2].flight_pairs
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = [exe.submit(check_couple_condition,
                                  [mat, [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB] +
                                   [fl.slot.index for fl in pairC]])
                       for pairA in fl_pair_a for pairB in fl_pair_b for pairC in fl_pair_c]
            # print(len([match for f in concurrent.futures.as_completed(results)
            #            for match in f.result() if len(f.result()) > 0]))

    else:
        for pairA in fl_pair_a:
            for pairB in fl_pair_b:
                for pairC in fl_pair_c:

                    if check_couple_condition(mat, [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB] +
                                                   [fl.slot.index for fl in pairC]):
                        matches.append([pairA, pairB, pairC])
    return matches


def run_triples_check(mat, airlines_triples, parallel=False):
    matches = []

    for air_triple in airlines_triples:
        match = air_triple_check(mat, air_triple, parallel)
        if len(match) > 0:
            matches += match
    return matches


def check_couple_in_pairs(mat, couple, airlines_pairs, paralle=False):
    matches = []
    other_airline = None

    for air_pair in airlines_pairs:
        if couple[0].airline.name == air_pair[0].name:
            other_airline = air_pair[1]
        elif couple[0].airline.name == air_pair[1].name:
            other_airline = air_pair[0]

        if other_airline is not None:
            for pair in other_airline.flight_pairs:
                if check_couple_condition(mat, [fl.slot.index for fl in couple] + [fl.slot.index for fl in pair]):
                    matches.append([couple, pair])

    return matches


def check_couple_in_triples(mat, couple, airlines_triples, paralle=False):
    matches = []
    other_airline_A = None
    other_airline_B = None

    for air_pair in airlines_triples:
        if couple[0].airline.name == air_pair[0].name:
            other_airline_A = air_pair[1]
            other_airline_B = air_pair[2]
        elif couple[0].airline.name == air_pair[1].name:
            other_airline_A = air_pair[0]
            other_airline_B = air_pair[2]
        elif couple[0].airline.name == air_pair[2].name:
            other_airline_A = air_pair[0]
            other_airline_B = air_pair[1]

        if other_airline_A is not None:
            for pairB in other_airline_A.flight_pairs:
                for pairC in other_airline_B.flight_pairs:

                    if check_couple_condition(mat, [fl.slot.index for fl in couple] + [fl.slot.index for fl in pairB] +
                                                   [fl.slot.index for fl in pairC]):
                        matches.append([couple, pairB, pairC])

    return matches